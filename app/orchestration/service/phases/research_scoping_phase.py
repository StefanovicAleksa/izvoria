# Path: app/orchestration/service/phases/research_scoping_phase.py

import concurrent.futures
import logging
import uuid
from typing import List, Optional, Tuple

from app.orchestration.utils.preload_slot import PreloadSlot
from app.core.config import settings
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType
from app.core.jobs.service.manager import JobManager
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.features.deep_research_report.domain.entities import CorpusDigest, Report
from app.features.deep_research_report.service.report_api import ReportAuthoringAPI
from app.orchestration.domain.entities import PhaseBatchResult, ResearchScopingJobResult
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase

logger = logging.getLogger(__name__)


class ResearchScopingPhase:
    """
    Wraps the single scoping call that turns a topic into a proposed outline.

    One model call, one job. The multi-stage run lives in DeepResearchReportPipeline; this Phase, like
    every other Phase, wraps exactly one logical step.
    """

    # One per request thread: concurrent requests share this Phase (see PreloadSlot)
    _preload_future = PreloadSlot()

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
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._loader)

    def execute(
            self,
            scoping_requests: List[Tuple[str, CorpusDigest, Optional[uuid.UUID]]],
    ) -> PhaseBatchResult:
        """
        Takes (topic, corpus_digest, session_id) tuples. In practice the pipeline always calls this with
        exactly one -- a person scopes one report at a time -- but the batch shape matches every other
        Phase, and the model lifecycle below is written to hold one residency across all of them anyway.
        """
        logger.info(f"Executing Research Scoping Phase for {len(scoping_requests)} topic(s)...")

        results: List[Tuple[ResearchScopingJobResult, Optional[Report]]] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded Report Authoring LLM.")
            except ModelManagerError as e:
                logger.error(f"Failed to load the Report Authoring LLM: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.RESEARCH_SCOPING,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading the Report Authoring LLM: {str(e)}"],
                )

        try:
            for topic, corpus, session_id in scoping_requests:
                request = JobRequest(
                    job_type=JobType.RESEARCH_SCOPING,
                    source_id=None,
                    input_params={"topic": topic},
                    progress_detail=f"Planning the report: {topic}",
                )

                try:
                    report: Report = self._job_manager.run_job(
                        request,
                        self._report_api.scope_plan,
                        topic=topic,
                        corpus=corpus,
                        session_id=session_id,
                        release_after=loaded_model is None,
                    )
                    results.append((
                        ResearchScopingJobResult(
                            report_id=report.id,
                            success=True,
                            section_count=len(report.plan.sections),
                            sub_question_count=len(report.plan.sub_questions),
                        ),
                        report,
                    ))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Scoping failed for topic '{topic}': {err_msg}")
                    results.append((
                        ResearchScopingJobResult(success=False, error_message=err_msg), None
                    ))
                    errors.append(f"Failed scoping for topic '{topic}': {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                logger.info("Unloading the Report Authoring LLM after scoping...")
                self._async_service.trigger_unload(
                    loaded_model=loaded_model,
                    loader=self._loader,
                    flush=True,
                )

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.RESEARCH_SCOPING,
            status=status,
            total_processed=len(scoping_requests),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors,
        )
