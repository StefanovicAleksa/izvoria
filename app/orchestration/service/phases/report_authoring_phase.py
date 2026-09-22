# Path: app/orchestration/service/phases/report_authoring_phase.py

import concurrent.futures
import logging
from dataclasses import dataclass
import uuid
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.jobs.domain.entities import CancellationToken, JobRequest
from app.core.jobs.domain.enums import JobType
from app.core.jobs.exceptions.execution import JobCancelled
from app.core.jobs.service.manager import JobManager
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.features.deep_research_report.domain.entities import (
    EvidenceEntry, FollowUpQuery, OutlineAssignment, OutlineSection, ReportSection, ResearchPlan,
)
from app.features.deep_research_report.domain.enums import SectionStatus
from app.features.deep_research_report.service.report_api import ReportAuthoringAPI
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.orchestration.domain.entities import (
    PhaseBatchResult, ReportOutlineJobResult, ReportSectionJobResult,
)
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Summary:
    """The executive summary, carried back in the phase result without touching the job counters."""
    report_id: uuid.UUID
    text: str


@dataclass(frozen=True)
class _GapStop:
    """
    Marker carried in the phase result when `stop_on_gaps` ended the run before writing.

    A private dataclass rather than a new orchestration entity: nothing outside this phase and its two
    accessors below ever needs to know the shape, and PhaseBatchResult.items is already List[Any].
    """
    report_id: uuid.UUID
    assignment: OutlineAssignment


class ReportAuthoringPhase:
    """
    Outline optimisation, every section write and the executive summary -- all inside ONE model residency.

    This is deliberately a single `execute_report()` rather than `execute_outline()` + `execute_sections()`.
    Each `execute` resolves a preload and unloads in its own `finally`, so splitting this in two would mean
    two `finally` unloads and two loads of a ~5.7 GB file. Across an eight-section report the difference
    between this shape and a per-section one is roughly eight heavy loads versus one -- minutes of pure
    model loading on a 3060. If you are tempted to split it for tidiness, don't: the VRAM-timeline test
    exists to catch exactly that change.

    Progress still ticks per unit: each section is its own `run_job`, so a caller sees "section 4 of 8"
    even though the model never moves.
    """

    def __init__(
            self,
            job_manager: JobManager,
            report_api: ReportAuthoringAPI,
            async_model_service: AsyncModelService,
            loader: IModelLoader,
    ):
        self._job_manager = job_manager
        self._report_api = report_api
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        # Discard any preload a previous call fired but nobody resolved. Without this, overwriting
        # `_preload_future` below makes the old one unreachable: the background worker still loads that
        # model, `discard_preload` can never be called on it, nothing releases it, and the next
        # `acquire_model(..., wait_for_clear=True)` parks forever with no exception and no way out but a
        # restart. Straight-line pipelines resolve every preload exactly once so this never fired; the
        # report mode's gather loop calls preload_model once per round, which makes it load-bearing.
        self.discard_preload()
        logger.info("Triggering background preload for the Report Authoring LLM...")
        manifest = ModelManifest(
            model_type=ModelType.REPORT_AUTHORING,
            model_path=settings.REPORT_AUTHORING_MODEL_PATH,
            required_vram_mb=settings.REPORT_AUTHORING_VRAM_REQUIREMENT_MB,
        )
        self._preload_future = self._async_service.trigger_preload(
            manifest=manifest,
            loader=self._loader,
        )

    def discard_preload(self) -> None:
        """Releases a preload that execute_report() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._loader)

    def execute_report(
            self,
            report_id: uuid.UUID,
            plan: ResearchPlan,
            evidence: List[EvidenceEntry],
            chunk_lookup: Dict[uuid.UUID, RetrievedChunk],
            pending_keys: Optional[List[str]] = None,
            optimize: bool = True,
            stop_on_gaps: bool = False,
            cancel_token: Optional[CancellationToken] = None,
    ) -> PhaseBatchResult:
        """
        Writes the report.

        `pending_keys` limits the run to sections still unwritten, which is what makes a resume cheap: it
        reuses the persisted bank and this one model load, and touches no retrieval model at all.
        `optimize=False` skips the outline pass on resume, since the assignments are already stored.

        `stop_on_gaps` is how the exhaustive mode's loop works: if the outline pass succeeds and leaves
        any section unsupported, the phase returns after assignment WITHOUT writing, and its `finally`
        releases the model. The caller is then free to gather again -- which it could not do while this
        phase held the model, because `acquire_model(wait_for_clear=True)` would park forever.

        Deliberately a flag rather than a callback. A callback invoked from in here would be handed the
        exact footgun this design exists to avoid: something that runs while the model is held and whose
        entire purpose is to decide "go gather more".

        Cancellation is checked between sections. A cancelled run keeps everything already written and
        stops -- the caller records the report as CANCELLED and it stays resumable.
        """
        logger.info(f"Executing Report Authoring Phase for report {report_id}...")

        results: List[Tuple[object, Optional[object]]] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        cancelled = False
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded Report Authoring LLM.")
            except ModelManagerError as e:
                logger.error(f"Failed to load the Report Authoring LLM: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.REPORT_SECTION,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading the Report Authoring LLM: {str(e)}"],
                )

        # Held for every call below. The feature API's own acquire() finds this exact manifest already
        # resident and returns it immediately, so nothing here reloads the file.
        hold = loaded_model is not None

        try:
            sections = list(plan.sections)

            assignment: Optional[OutlineAssignment] = None
            outline_succeeded = False

            # ---------- 1. Outline optimisation ----------
            if optimize:
                try:
                    assignment = self._job_manager.run_job(
                        JobRequest(
                            job_type=JobType.RESEARCH_OUTLINE,
                            source_id=None,
                            input_params={"report_id": str(report_id)},
                            progress_detail="Assigning evidence to the outline",
                        ),
                        self._report_api.optimize_outline,
                        cancel_token=cancel_token,
                        plan=plan,
                        evidence=evidence,
                        release_after=not hold,
                    )
                    sections = list(assignment.sections)
                    outline_succeeded = True
                    covered = sum(1 for s in sections if s.evidence_doc_numbers)
                    results.append((
                        ReportOutlineJobResult(
                            report_id=report_id,
                            success=True,
                            covered_section_count=covered,
                            section_count=len(sections),
                        ),
                        sections,
                    ))
                    success_count += 1

                except JobCancelled:
                    raise
                except Exception as e:
                    # A failed outline pass is recoverable: fall back to the approved outline with whatever
                    # assignments it already carries, rather than losing a whole gathered bank.
                    err_msg = str(e)
                    logger.error(f"Outline optimisation failed for report {report_id}: {err_msg}")
                    results.append((
                        ReportOutlineJobResult(
                            report_id=report_id, success=False, section_count=len(sections),
                            error_message=err_msg,
                        ),
                        None,
                    ))
                    errors.append(f"Outline optimisation failed: {err_msg}")
                    failed_count += 1

            # ---------- 1b. Stop here if the caller wants another gathering round ----------
            # Gated on the outline job having SUCCEEDED. A failed one falls back to `plan.sections`, whose
            # evidence_doc_numbers default to [] -- so a single bad JSON parse would look like "every
            # section is a gap" and burn the entire round budget re-gathering evidence that was fine.
            if stop_on_gaps and outline_succeeded and assignment is not None and assignment.uncovered:
                logger.info(
                    f"Report {report_id}: {len(assignment.uncovered)} section(s) unsupported; stopping "
                    f"before writing so the caller can gather again."
                )
                return PhaseBatchResult(
                    # COMPLETED, not FAILED: the pipeline raises WorkflowStateError on FAILED, which would
                    # abort the run instead of looping.
                    phase=PipelinePhase.REPORT_SECTION,
                    status=PhaseStatus.COMPLETED,
                    total_processed=len(results),
                    successful_count=success_count,
                    failed_count=failed_count,
                    items=results + [(_GapStop(report_id=report_id, assignment=assignment), None)],
                    errors=errors,
                )

            # ---------- 2. Section writing ----------
            writable = [s for s in sections if pending_keys is None or s.key in set(pending_keys)]
            outline_context = self._outline_context(sections)
            written: List[ReportSection] = []

            for position, section in enumerate(writable, start=1):
                try:
                    report_section: ReportSection = self._job_manager.run_job(
                        JobRequest(
                            job_type=JobType.RESEARCH_SECTION,
                            source_id=None,
                            input_params={"report_id": str(report_id), "section_key": section.key},
                            progress_detail=(
                                f"Writing section {position}/{len(writable)}: {section.title}"
                            ),
                        ),
                        self._report_api.write_section,
                        cancel_token=cancel_token,
                        report_id=report_id,
                        topic=plan.topic,
                        section=section,
                        evidence=evidence,
                        chunk_lookup=chunk_lookup,
                        outline_context=outline_context,
                        release_after=not hold,
                    )
                    written.append(report_section)
                    results.append((
                        ReportSectionJobResult(
                            report_id=report_id,
                            section_key=section.key,
                            success=True,
                            status=report_section.status.value,
                            cited_count=len(report_section.cited_doc_numbers),
                        ),
                        report_section,
                    ))
                    success_count += 1

                except JobCancelled:
                    logger.info(
                        f"Report {report_id} cancelled after {len(written)} of {len(writable)} section(s); "
                        f"keeping what is written."
                    )
                    cancelled = True
                    break

                except Exception as e:
                    # One section failing must not lose the other seven. It is recorded as failed and the
                    # run continues; the report simply reports that section as not covered.
                    err_msg = str(e)
                    logger.error(f"Section {section.key!r} of report {report_id} failed: {err_msg}")
                    results.append((
                        ReportSectionJobResult(
                            report_id=report_id, section_key=section.key, success=False,
                            error_message=err_msg,
                        ),
                        None,
                    ))
                    errors.append(f"Section {section.key} failed: {err_msg}")
                    failed_count += 1

            # ---------- 3. Executive summary, on the SAME residency ----------
            # Previously its own phase method, which reloaded the 5.7 GB model for one short call. Skipped
            # on cancellation so the cancel path reaches _finish_cancelled without waiting on it.
            if not cancelled:
                summary = self._summarize(report_id, plan.topic, hold)
                if summary:
                    # Kept OUT of success_count/failed_count on purpose: a cosmetic summary failure must
                    # never flip the phase to FAILED and lose a complete report.
                    results.append((_Summary(report_id=report_id, text=summary), None))

        finally:
            if loaded_model:
                logger.info("Unloading the Report Authoring LLM after writing the report...")
                self._async_service.trigger_unload(
                    loaded_model=loaded_model,
                    loader=self._loader,
                    flush=True,
                )

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.REPORT_SECTION,
            status=status,
            total_processed=len(results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors + (["Cancelled by the user."] if cancelled else []),
        )

    def _summarize(self, report_id: uuid.UUID, topic: str, hold: bool) -> str:
        """
        The executive summary, written on the residency that just wrote the sections.

        Reads the written sections back from the DATABASE rather than from this run's results, because a
        resumed report's summary must cover all eight sections, not just the two the resume wrote.
        """
        try:
            stored = self._report_api.get_report(report_id)
            written = stored.written_sections if stored else []
            if not written:
                return ""

            return self._job_manager.run_job(
                JobRequest(
                    job_type=JobType.RESEARCH_SUMMARY,
                    source_id=None,
                    input_params={"report_id": str(report_id)},
                    progress_detail="Writing the executive summary",
                ),
                self._report_api.write_summary,
                topic=topic,
                sections=written,
                release_after=not hold,
            )
        except Exception as e:
            # A report without a summary is still a report; losing the sections over it would not be.
            logger.error(f"Executive summary failed for report {report_id}: {e}")
            return ""

    @staticmethod
    def _outline_context(sections: List[OutlineSection]) -> str:
        """The surrounding headings, so a section knows what the others already cover and does not repeat it."""
        return "\n".join(f"{index}. {section.title}" for index, section in enumerate(sections, start=1))

    @staticmethod
    def written_sections(result: PhaseBatchResult) -> List[ReportSection]:
        """The sections this phase actually wrote, in order, for the summary and the renderer."""
        return [
            payload for _, payload in result.items
            if isinstance(payload, ReportSection) and payload.status == SectionStatus.WRITTEN
        ]

    @staticmethod
    def was_cancelled(result: PhaseBatchResult) -> bool:
        return "Cancelled by the user." in result.errors

    @staticmethod
    def stopped_for_gaps(result: PhaseBatchResult) -> bool:
        """True when the run stopped after assignment so the caller could gather another round."""
        return any(isinstance(job_result, _GapStop) for job_result, _ in result.items)

    @staticmethod
    def follow_ups(result: PhaseBatchResult) -> List[FollowUpQuery]:
        """The queries to run next round. Empty unless the run stopped for gaps."""
        for job_result, _ in result.items:
            if isinstance(job_result, _GapStop):
                return list(job_result.assignment.follow_ups)
        return []

    @staticmethod
    def summary(result: PhaseBatchResult) -> str:
        """The executive summary written on this run's residency, or "" if none was."""
        for job_result, _ in result.items:
            if isinstance(job_result, _Summary):
                return job_result.text
        return ""
