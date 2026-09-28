# Path: app/features/deep_research_report/service/report_api.py

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from app.core.anchors.domain.enums import AnchorKind
from app.core.config import settings
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, EvidenceEntry, FollowUpQuery, OutlineAssignment, OutlineSection, QueryAttempt,
    Report, ReportSection, ReportSummary, ResearchPlan,
)
from app.features.deep_research_report.domain.enums import ReportMode, ReportStatus, SectionStatus
from app.features.deep_research_report.domain.interfaces import (
    IReportFileStore, IReportLlmAdapter, IReportRepository,
)
from app.features.deep_research_report.utils import markdown_renderer
from app.features.deep_research_report.utils.error_mapper import map_report_error
from app.features.deep_research_report.utils.evidence_bank import (
    build_bank_index, build_section_digest, chunks_for_section, pack_section_evidence,
)
from app.features.deep_research_report.utils.query_log import (
    already_asked, format_attempts, normalise_query,
)
from app.features.qa_generation.utils.citation_parser import (
    build_numbered_context_block, normalize_citation_tags,
)
from app.features.qa_generation.utils.context_packer import has_document_chunks
from app.features.rag_retrieval.domain.entities import RetrievedChunk

logger = logging.getLogger(__name__)

_DOC_TAG = re.compile(r"\[Doc (\d+)]")
# A heading the model emitted at the very top of a section body.
_LEADING_HEADING = re.compile(r"\A\s*#{1,6}[^\n]*\n+")


class ReportAuthoringAPI:
    """
    The Deep Research Report feature's public entrypoint: scoping, outline optimisation, section writing,
    summarising, rendering and persistence.

    Every model-facing method takes `release_after`, exactly as the chat-path APIs do. A report run sets it
    to False for everything after the first call, which is what lets one acquisition of the ~5.7 GB model
    cover outline optimisation, all eight section writes and the summary. The orchestrator returns an
    already-resident model of the same manifest immediately, so these calls cost nothing extra.
    """

    def __init__(
            self,
            adapter: IReportLlmAdapter,
            orchestrator: ModelManagerOrchestrator,
            repository: IReportRepository,
            file_store: IReportFileStore,
    ):
        self._adapter = adapter
        self._orchestrator = orchestrator
        self._repository = repository
        self._file_store = file_store
        self._manifest = ModelManifest(
            model_type=ModelType.REPORT_AUTHORING,
            model_path=settings.REPORT_AUTHORING_MODEL_PATH,
            required_vram_mb=settings.REPORT_AUTHORING_VRAM_REQUIREMENT_MB,
        )

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    def _acquire(self) -> LoadedModel:
        return self._orchestrator.acquire_model(self._manifest, self._adapter, wait_for_clear=True)

    def _release(self, model: LoadedModel, release_after: bool) -> None:
        if release_after:
            self._orchestrator.release_model(model, self._adapter, flush=True)

    # ==========================================
    # 1. SCOPING
    # ==========================================

    def scope_plan(
            self,
            topic: str,
            corpus: CorpusDigest,
            session_id: Optional[uuid.UUID] = None,
            release_after: bool = True,
    ) -> Report:
        """
        Proposes an outline against what the corpus actually holds, and persists it as AWAITING_APPROVAL.

        Returns a `Report`, not a bare plan, because the caller needs the `report_id` to hand back with the
        approved outline -- the two-call shape is what keeps a 30-minute run from starting on a misread topic.
        """
        logger.info(f"Scoping a research plan for topic: '{topic}'")
        try:
            loaded_model = self._acquire()
            try:
                plan = self._adapter.scope(loaded_model, topic, corpus)
            finally:
                self._release(loaded_model, release_after)

            report = self._repository.create_report(
                topic=topic, plan=plan, corpus=corpus, session_id=session_id
            )
            logger.info(
                f"Proposed {len(plan.sections)} section(s) and {len(plan.sub_questions)} sub-question(s) "
                f"for report {report.id}."
            )
            return report

        except Exception as e:
            raise map_report_error(e, context=f"Scoping a research plan for '{topic}'") from e

    def approve_plan(self, report_id: uuid.UUID, plan: ResearchPlan) -> Report:
        """Stores the user's approved outline and moves the report to GATHERING. No model involved."""
        try:
            return self._repository.replace_plan(report_id, plan)
        except Exception as e:
            raise map_report_error(e, context=f"Approving the plan of report {report_id}") from e

    # ==========================================
    # 2. EVIDENCE BANK
    # ==========================================

    def save_evidence_bank(
            self,
            report_id: uuid.UUID,
            evidence: List[EvidenceEntry],
            swept_source_count: int,
            scrub_enabled: bool = False,
            mode: ReportMode = ReportMode.DEEP,
            rounds_used: int = 1,
    ) -> None:
        """Commits the numbered bank. After this a resume needs no retrieval model at all."""
        try:
            self._repository.save_evidence_bank(
                report_id, evidence, swept_source_count, scrub_enabled, mode, rounds_used
            )
            logger.info(f"Persisted {len(evidence)} evidence entries for report {report_id}.")
        except Exception as e:
            raise map_report_error(e, context=f"Saving the evidence bank of report {report_id}") from e

    # ==========================================
    # 3. OUTLINE OPTIMISATION
    # ==========================================

    def optimize_outline(
            self,
            plan: ResearchPlan,
            evidence: List[EvidenceEntry],
            release_after: bool = False,
            asked: Optional[List[QueryAttempt]] = None,
    ) -> OutlineAssignment:
        """
        Assigns the gathered bank to the outline's sections, and says what to search for next.

        The planner reads the bank *index* -- one line per entry -- not the transcripts. A 200-entry bank
        is then about 12K tokens, which is what allows this call and every section write to share one
        loaded model instead of alternating between a planner and a writer.

        Every uncovered section comes back with a follow-up query, whether or not the model produced one:
        an exhaustive run's whole second round depends on having queries to run, and the case that most
        needs one -- a bank so empty the model is never called at all -- is exactly the case the model
        cannot answer.

        `asked` is what the run has already searched for. It is applied twice: the model is shown it, so
        its follow-up is a new question rather than a paraphrase, and every follow-up is checked against
        it afterwards, because the fallback is derived from a section's unchanging title and brief and
        would otherwise hand back the previous round's query verbatim. A round that can only re-ask what
        has already been asked cannot find anything, and costs an embedder, a reranker and a ~5.7 GB
        model load to prove it. Returning no follow-ups is how this call says "stop".
        """
        if not evidence:
            logger.info(f"Report {plan.report_id} gathered no evidence; every section is uncovered.")
            uncovered = [s.model_copy(update={"evidence_doc_numbers": []}) for s in plan.sections]
            return OutlineAssignment(
                sections=uncovered,
                follow_ups=self._fresh_follow_ups(uncovered, [], asked),
            )

        try:
            loaded_model = self._acquire()
            try:
                assignment = self._adapter.optimize_outline(
                    loaded_model, plan, build_bank_index(evidence), format_attempts(asked or []),
                    has_documents=any(entry.anchor_kind is not AnchorKind.TIME for entry in evidence)
                )
            finally:
                self._release(loaded_model, release_after)

            valid_numbers = {e.doc_number for e in evidence}
            # A hallucinated document number would become a citation pointing at nothing, so the
            # assignment is filtered against the bank rather than trusted.
            cleaned = [
                section.model_copy(update={
                    "evidence_doc_numbers": [n for n in section.evidence_doc_numbers if n in valid_numbers]
                })
                for section in assignment.sections
            ]

            self._repository.save_section_assignments(plan.report_id, cleaned)
            covered = sum(1 for s in cleaned if s.evidence_doc_numbers)
            logger.info(
                f"Outline optimisation covered {covered}/{len(cleaned)} section(s) of report {plan.report_id}."
            )

            result = OutlineAssignment(sections=cleaned, follow_ups=assignment.follow_ups)
            return result.model_copy(update={
                "follow_ups": self._fresh_follow_ups(result.uncovered, result.follow_ups, asked)
            })

        except Exception as e:
            raise map_report_error(e, context=f"Optimising the outline of report {plan.report_id}") from e

    @staticmethod
    def _fresh_follow_ups(
            uncovered: List[OutlineSection],
            existing: List[FollowUpQuery],
            asked: Optional[List[QueryAttempt]],
    ) -> List[FollowUpQuery]:
        """
        Every uncovered section's next query, minus anything this run has already searched for.

        The model's own follow-ups are filtered first, then each remaining gap is filled from the section's
        title and brief -- deterministically, with no model call, the same shape
        `outline_validator._fill_empty_sections` uses. A generated query is worse than a model-written one,
        which is why it only fills gaps the model left, but a worse query beats no second round at all.

        The filter is what stops the loop spinning. Neither the model nor the fallback knows what earlier
        rounds tried: the fallback is a pure function of a title and brief that do not change between
        rounds, so an uncovered section the model writes nothing for regenerates the identical query every
        round. Dropping it here means an exhausted section contributes no query, and a round with no
        queries left is a round the caller skips instead of paying for.
        """
        seen = already_asked(asked or [])
        filled: List[FollowUpQuery] = []

        for follow_up in existing:
            key = normalise_query(follow_up.text)
            if not key or key in seen:
                continue
            seen.add(key)
            filled.append(follow_up)

        covered_keys = {f.section_key for f in filled}
        for section in uncovered:
            if section.key in covered_keys:
                continue
            text = f"{section.title.strip()}. {section.brief.strip()}".strip().rstrip(".") + "."
            key = normalise_query(text)
            if not key or key in seen:
                continue
            seen.add(key)
            filled.append(FollowUpQuery(section_key=section.key, text=text))

        return filled

    # ==========================================
    # 4. SECTION WRITING
    # ==========================================

    def write_section(
            self,
            report_id: uuid.UUID,
            topic: str,
            section: OutlineSection,
            evidence: List[EvidenceEntry],
            chunk_lookup: Dict[uuid.UUID, RetrievedChunk],
            outline_context: str = "",
            release_after: bool = False,
    ) -> ReportSection:
        """
        Writes one section from its own evidence only, validates its citations, and persists it.

        Two guarantees enforced here rather than asked for in the prompt:

        1. The writer sees only `raw_text`, via `build_numbered_context_block` -- the same function the chat
           path uses, and the same reason: `text_payload` carries generated situating context, and an answer
           grounded in that would still come out carrying a valid-looking citation.
        2. Citations are normalised against THIS section's allowed numbers. Without that restriction a
           section could cite a document it was never shown, because the numbers are report-wide.

        A section that ends up citing nothing becomes SKIPPED_NO_EVIDENCE and is reported as uncovered --
        never uncited prose presented as a finding.
        """
        entries_by_number = {e.doc_number: e for e in evidence}
        pairs = chunks_for_section(section.evidence_doc_numbers, entries_by_number, chunk_lookup)

        if not pairs:
            logger.info(f"Section {section.key!r} of report {report_id} has no resolvable evidence; skipping.")
            return self._persist_section(report_id, section, "", [], SectionStatus.SKIPPED_NO_EVIDENCE)

        try:
            loaded_model = self._acquire()
            try:
                has_documents = has_document_chunks([chunk for _, chunk in pairs])
                budget = self._adapter.get_section_token_budget(loaded_model, section, topic,
                                                                has_documents=has_documents)
                packed = pack_section_evidence(
                    pairs, budget, lambda text: self._adapter.count_tokens(loaded_model, text)
                )
                if not packed:
                    logger.warning(
                        f"No evidence for section {section.key!r} fits a {budget}-token budget; skipping."
                    )
                    return self._persist_section(
                        report_id, section, "", [], SectionStatus.SKIPPED_NO_EVIDENCE
                    )

                context = build_numbered_context_block(packed)
                allowed = [number for number, _ in packed]

                body = self._adapter.write_section(
                    model=loaded_model,
                    topic=topic,
                    section=section,
                    evidence_block=context.formatted_text,
                    allowed_doc_numbers=allowed,
                    outline_context=outline_context,
                    has_documents=has_documents,
                )
            finally:
                self._release(loaded_model, release_after)

            return self._persist_written_section(report_id, section, body, allowed)

        except Exception as e:
            raise map_report_error(
                e, context=f"Writing section {section.key} of report {report_id}"
            ) from e

    @staticmethod
    def _strip_leading_heading(body: str) -> str:
        """
        Drops a heading the model repeated at the top of its own section.

        The renderer owns section headings, and the prompt asks for the body only -- but models emit the
        heading anyway often enough that the rendered report ends up showing it twice. Only a heading at
        the very start is removed; sub-headings inside the body are the model's own structure and stay.
        """
        return _LEADING_HEADING.sub("", body, count=1)

    def _persist_written_section(
            self,
            report_id: uuid.UUID,
            section: OutlineSection,
            body: str,
            allowed: List[int],
    ) -> ReportSection:
        stripped = self._strip_leading_heading(body.strip()).strip()
        cleaned = normalize_citation_tags(stripped, len(allowed), valid_numbers=set(allowed))
        cited = sorted(set(self._cited_numbers(cleaned)))

        if not cited:
            logger.info(
                f"Section {section.key!r} of report {report_id} cites nothing after a retry; recording it "
                f"as not covered rather than shipping uncited prose."
            )
            return self._persist_section(report_id, section, "", [], SectionStatus.SKIPPED_NO_EVIDENCE)

        return self._persist_section(report_id, section, cleaned, cited, SectionStatus.WRITTEN)

    @staticmethod
    def _cited_numbers(text: str) -> List[int]:
        return [int(m.group(1)) for m in _DOC_TAG.finditer(text)]

    def _persist_section(
            self,
            report_id: uuid.UUID,
            section: OutlineSection,
            body: str,
            cited: List[int],
            status: SectionStatus,
    ) -> ReportSection:
        written = ReportSection(
            key=section.key,
            title=section.title,
            order_index=section.order_index,
            status=status,
            markdown_body=body,
            cited_doc_numbers=cited,
            written_at=datetime.now(timezone.utc),
        )
        # Committed per section, so a crash or a cancel costs at most the section in flight.
        self._repository.save_section(report_id, written)
        return written

    # ==========================================
    # 5. EXECUTIVE SUMMARY + RENDER
    # ==========================================

    def write_summary(
            self,
            topic: str,
            sections: List[ReportSection],
            release_after: bool = True,
    ) -> str:
        """
        Writes the summary from the sections' own prose, restricted to the numbers they already cite.

        The summary is the one place in a report that synthesises across sections, which makes it the
        easiest place to assert a connection no single source supports. Both restrictions exist for that:
        it never sees raw evidence, and its citations are validated against the body's.
        """
        written = [s for s in sections if s.status == SectionStatus.WRITTEN and s.markdown_body.strip()]
        if not written:
            return ""

        allowed = sorted({n for s in written for n in s.cited_doc_numbers})

        try:
            loaded_model = self._acquire()
            try:
                summary = self._adapter.write_summary(
                    model=loaded_model,
                    topic=topic,
                    section_digest=build_section_digest([(s.title, s.markdown_body) for s in written]),
                    allowed_doc_numbers=allowed,
                )
            finally:
                self._release(loaded_model, release_after)

            return normalize_citation_tags(summary.strip(), len(allowed), valid_numbers=set(allowed))

        except Exception as e:
            raise map_report_error(e, context=f"Writing the executive summary of '{topic}'") from e

    def render_and_finalize(self, report_id: uuid.UUID, executive_summary: str) -> Report:
        """
        Renders the report to Markdown, writes it to disk and marks it COMPLETED.

        Rendered from the persisted report rather than from in-memory state, so the file on disk and the
        row in the database can never disagree about what the report says.
        """
        try:
            report = self._repository.get_report(report_id)
            if report is None:
                raise map_report_error(
                    ValueError(f"Report {report_id} vanished before rendering"), context="Rendering"
                )

            with_summary = report.model_copy(update={"executive_summary": executive_summary})
            markdown = markdown_renderer.render(with_summary)
            path = self._file_store.write_markdown(report_id, markdown)
            return self._repository.finalize(report_id, executive_summary, path)

        except Exception as e:
            raise map_report_error(e, context=f"Rendering and finalizing report {report_id}") from e

    def render_partial(self, report_id: uuid.UUID) -> Tuple[Optional[Report], Optional[str]]:
        """
        Renders whatever a cancelled or failed run produced, without marking it COMPLETED.

        A stopped report is still worth reading: the sections that were written are cited and valid, and a
        person who cancelled after twenty minutes should get those twenty minutes of work rather than an
        empty row. The file is written; the status is left as-is so `resume` still knows what is pending.

        Returns `(None, None)` for a report that cannot be read at all -- this runs on the cancel path,
        where raising would turn a clean stop into a crash, so the caller decides what an unreadable
        report means. The report is optional in the return type for that reason.
        """
        try:
            report = self._repository.get_report(report_id)
            if report is None or not report.written_sections:
                return report, None

            path = self._file_store.write_markdown(report_id, markdown_renderer.render(report))
            return report, path

        except Exception as e:
            logger.error(f"Could not render the partial report {report_id}: {e}")
            # Re-read rather than reuse `report`, which may be unbound if the read itself was what
            # failed. Guarded, because a repository that just failed is likely to fail again, and an
            # unmapped repository exception escaping here would leave the feature boundary as a raw
            # SQLAlchemy error -- and would do it on the cancel path, of all places.
            try:
                return self._repository.get_report(report_id), None
            except Exception as read_error:
                logger.error(f"Could not read report {report_id} back either: {read_error}")
                return None, None

    # ==========================================
    # 6. READS / STATUS
    # ==========================================

    def get_report(self, report_id: uuid.UUID) -> Optional[Report]:
        try:
            return self._repository.get_report(report_id)
        except Exception as e:
            raise map_report_error(e, context=f"Loading report {report_id}") from e

    def list_reports(self, session_id: Optional[uuid.UUID] = None, limit: int = 50) -> List[ReportSummary]:
        try:
            return self._repository.list_reports(session_id, limit)
        except Exception as e:
            raise map_report_error(e, context="Listing reports") from e

    def set_status(
            self,
            report_id: uuid.UUID,
            status: ReportStatus,
            error_message: Optional[str] = None,
    ) -> None:
        try:
            self._repository.set_status(report_id, status, error_message)
        except Exception as e:
            raise map_report_error(e, context=f"Setting the status of report {report_id}") from e
