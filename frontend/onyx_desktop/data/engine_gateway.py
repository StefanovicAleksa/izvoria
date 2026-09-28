# Canvas: desktop_engine_gateway_v1
# Path: frontend/onyx_desktop/data/engine_gateway.py

"""
The only module in the desktop app that imports `app.*`.

Everything above this file speaks in frozen dataclasses from onyx_desktop.domain; everything
below it is the appliance. That boundary is what lets the UI run and be tested against
StubEngineGateway with no GPU, database or model weights present.

The engine imports are deliberately deferred into `boot()` rather than done at module import:
importing `app.*` pulls in torch and CUDA (seconds, hundreds of MB) and reads
`os.environ["ENV"]` at `app.core.config` import time. Deferring keeps `--stub` startup instant
and keeps a mis-set ENV from being baked in before main.py can fix it.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from onyx_desktop.domain.entities import (
    ChatMessage,
    Evidence,
    IngestionSummary,
    ProgressTick,
    ReasoningStep,
    ReportDocument,
    ReportOutlineDraft,
    ReportPlanDraft,
    SearchHit,
    SearchOutcome,
)
from onyx_desktop.domain.enums import (
    ChatMode,
    IngestMode,
    JobProgressState,
    MediaKind,
    MessageRole,
    ReportMode,
    ReportRunStatus,
    SearchMode,
    TurnOutcome,
)
from onyx_desktop.domain.interfaces import IEngineGateway
from onyx_desktop.exceptions.gateway import ApplianceOfflineError, InvalidRequestError
from onyx_desktop.utils.error_mapper import map_engine_error

logger = logging.getLogger(__name__)

# Populated by _import_engine(); module-level so the cost is paid once per process.
_engine_modules: Dict[str, Any] = {}


def _import_engine() -> Dict[str, Any]:
    """Imports the engine on first use and caches the handful of symbols the gateway needs."""
    if _engine_modules:
        return _engine_modules

    from app.engine.service.bootstrap import OnyxBootstrapper
    from app.features.rag_retrieval.domain.entities import SearchQuery
    from app.features.qa_generation.utils.citation_parser import normalize_citation_tags
    from app.features.deep_research_report.domain.entities import OutlineSection, SubQuestion
    from app.features.deep_research_report.utils import markdown_renderer
    from app.core.anchors.utils.formatting import format_location

    _engine_modules.update(
        bootstrapper=OnyxBootstrapper,
        SearchQuery=SearchQuery,
        normalize_citation_tags=normalize_citation_tags,
        # Report mode: rebuilding an edited plan needs the engine's own entities, and rendering needs
        # the renderer's link mode. Every symbol used anywhere in this file must be registered here --
        # a missing one only fails at call time, with a KeyError the error mapper cannot name.
        OutlineSection=OutlineSection,
        SubQuestion=SubQuestion,
        markdown_renderer=markdown_renderer,
        format_location=format_location,
    )
    return _engine_modules


class OnyxEngineGateway(IEngineGateway):
    """
    Drives OnyxEngine in-process. Every method is called from the single engine worker thread;
    none of them is safe to call concurrently, which is exactly how the bridge uses them.
    """

    def __init__(self) -> None:
        self._engine = None
        self._progress: Optional[Callable[[ProgressTick], None]] = None

    # ---------- lifecycle ----------

    def boot(self) -> None:
        modules = _import_engine()
        try:
            self._engine = modules["bootstrapper"].build(progress_listener=self._on_job_progress)
            self._engine.start()
        except Exception as exc:
            raise map_engine_error(exc, context="Starting the appliance")

    def shutdown(self) -> None:
        if self._engine is None:
            return
        try:
            self._engine.stop()
        except Exception as exc:
            # Shutdown failures must not prevent the window from closing; log and move on.
            logger.error(f"Engine shutdown failed: {exc}")
        finally:
            self._engine = None

    def is_ingesting(self) -> bool:
        if self._engine is None:
            return False
        try:
            return self._engine.get_state().value == "ingesting"
        except Exception:
            return False

    def set_progress_listener(self, listener: Optional[Callable[..., None]]) -> None:
        self._progress = listener

    # ---------- live inference ----------

    def chat(self, prompt: str, session_id: uuid.UUID, mode: ChatMode, limit: int) -> ChatMessage:
        engine = self._require_engine()
        modules = _import_engine()
        try:
            query = modules["SearchQuery"](prompt=prompt, limit=limit)
            if mode is ChatMode.REASONING:
                return self._map_reasoned(engine.chat_reasoned(session_id, query))
            response = engine.chat(query, session_id=session_id)
            return self._map_qa_response(response)
        except Exception as exc:
            raise map_engine_error(exc, context="Chat")

    def search(self, prompt: str, mode: SearchMode, limit: int) -> SearchOutcome:
        engine = self._require_engine()
        modules = _import_engine()
        try:
            query = modules["SearchQuery"](prompt=prompt, limit=limit)
            result = (
                engine.search_verified(query)
                if mode is SearchMode.VERIFIED
                else engine.search_standard(query)
            )
            hits = tuple(
                self._map_hit(rank, chunk)
                for rank, chunk in enumerate(result.results, start=1)
            )
            return SearchOutcome(
                query=result.query,
                hits=hits,
                execution_time_ms=result.execution_time_ms,
            )
        except Exception as exc:
            raise map_engine_error(exc, context=f"{mode.value.capitalize()} search")

    # ==========================================
    # RESEARCH REPORTS
    # ==========================================

    def propose_report_plan(self, topic: str, mode: ReportMode) -> ReportPlanDraft:
        engine = self._require_engine()
        try:
            proposal = engine.propose_research_plan(topic)
            plan = proposal.plan
            return ReportPlanDraft(
                report_id=proposal.report_id,
                topic=plan.topic,
                scope_note=plan.scope_note,
                sections=tuple(
                    ReportOutlineDraft(
                        key=section.key,
                        title=section.title,
                        brief=section.brief,
                        questions=tuple(q.text for q in plan.sub_questions_for(section.key)),
                    )
                    for section in plan.sections
                ),
                indexed_source_count=proposal.corpus.indexed_source_count,
                indexed_chunk_count=proposal.corpus.indexed_chunk_count,
                expected_job_count=proposal.expected_job_count,
                estimated_minutes=proposal.estimated_minutes,
            )
        except Exception as exc:
            raise map_engine_error(exc, context="Planning a research report")

    def run_report(self, plan: ReportPlanDraft, mode: ReportMode) -> ReportDocument:
        engine = self._require_engine()
        try:
            stored = engine.get_report(plan.report_id)
            if stored is None:
                raise InvalidRequestError("That research plan is no longer available; scope it again.")

            approved = self._apply_edits(stored.plan, plan)
            if mode is ReportMode.EXHAUSTIVE:
                report = engine.execute_iterative_research_report(plan.report_id, approved)
            else:
                report = engine.execute_research_report(plan.report_id, approved)
            return self._map_report(report)
        except Exception as exc:
            raise map_engine_error(exc, context="Running a research report")

    def request_report_cancel(self, report_id: uuid.UUID) -> bool:
        """
        Called on the GUI thread, bypassing the worker on purpose -- see IEngineGateway.

        Never raises: a Stop button that throws while the run keeps going is worse than one that
        reports it could not stop.
        """
        if self._engine is None:
            return False
        try:
            return bool(self._engine.cancel_research_report(report_id))
        except Exception as exc:
            logger.error(f"Could not cancel report {report_id}: {exc}")
            return False

    def load_report(self, report_id: uuid.UUID) -> Optional[ReportDocument]:
        engine = self._require_engine()
        try:
            report = engine.get_report(report_id)
            return self._map_report(report) if report is not None else None
        except Exception as exc:
            raise map_engine_error(exc, context="Loading a research report")

    @staticmethod
    def _apply_edits(stored_plan: Any, draft: ReportPlanDraft) -> Any:
        """
        Rebuilds the engine's plan from the user's edited draft.

        Section keys are carried through, not regenerated: the engine re-attaches evidence by key across
        plan revisions, so a reordered or renamed section keeps its identity. A section the user deleted
        simply does not appear, and its sub-questions go with it.
        """
        modules = _import_engine()
        by_key = {section.key: section for section in stored_plan.sections}

        sections = []
        questions = []
        for order_index, edited in enumerate(draft.sections):
            original = by_key.get(edited.key)
            base = original if original is not None else None
            sections.append(
                (base.model_copy(update={"title": edited.title, "brief": edited.brief,
                                         "order_index": order_index})
                 if base is not None else
                 modules["OutlineSection"](key=edited.key or f"u{order_index + 1}",
                                           title=edited.title, brief=edited.brief,
                                           order_index=order_index))
            )
            for text in edited.questions:
                if text.strip():
                    questions.append(modules["SubQuestion"](
                        text=text.strip(), section_key=sections[-1].key,
                        order_index=len(questions),
                    ))

        return stored_plan.model_copy(update={"sections": sections, "sub_questions": questions})

    def _map_report(self, report: Any) -> ReportDocument:
        """
        Renders the engine's report with clickable citations and hydrates the evidence behind them.

        `link_citations=True` is what turns `[Doc 7]` into `[Doc 7](onyx://doc/7)`; the file on disk keeps
        plain tags, so what the user saves is not littered with an internal URL scheme.
        """
        modules = _import_engine()
        markdown = modules["markdown_renderer"].render(report, link_citations=True)

        evidence: List[Evidence] = []
        try:
            cited = set(report.cited_doc_numbers)
            entries = [e for e in report.evidence if e.doc_number in cited]
            chunks = {c.chunk_id: c for c in self._engine.get_chunks([e.chunk_id for e in entries])}
            for entry in entries:
                chunk = chunks.get(entry.chunk_id)
                evidence.append(Evidence(
                    doc_number=entry.doc_number,
                    chunk_id=entry.chunk_id,
                    source_id=entry.source_id,
                    start_sec=entry.start_sec,
                    end_sec=entry.end_sec,
                    # A re-ingest can delete the chunk; the bank's stored digest still names the source.
                    excerpt=(chunk.raw_text if chunk is not None else entry.digest) or "",
                    speakers=tuple(entry.speaker_labels),
                    clip_path=getattr(chunk, "physical_path", None) if chunk is not None else None,
                    media_kind=self._media_kind(
                        getattr(chunk, "physical_path", None) if chunk is not None else None,
                        getattr(chunk, "extracted_file_id", None) if chunk is not None else None,
                        self._is_document(entry),
                        self._is_image(entry),
                    ),
                    location=self._location(entry),
                ))
        except Exception as exc:
            # A report whose citations cannot be hydrated is still worth reading.
            logger.warning(f"Could not hydrate report evidence: {exc}")

        return ReportDocument(
            report_id=report.id,
            topic=report.topic,
            markdown=markdown,
            status=ReportRunStatus(report.status.value),
            mode=ReportMode(report.mode.value),
            evidence=tuple(evidence),
            markdown_path=report.markdown_path,
            section_count=len(report.written_sections),
            uncovered_count=len(report.uncovered_sections),
            rounds_used=report.rounds_used,
            indexed_source_count=report.indexed_source_count,
            swept_source_count=report.swept_source_count,
        )

    def hydrate_evidence(self, chunk_ids: List[uuid.UUID]) -> List[Evidence]:
        engine = self._require_engine()
        try:
            chunks = engine.get_chunks(list(chunk_ids))
            # get_chunks collapses duplicates and silently drops ids that no longer exist, so
            # the result is keyed by chunk_id rather than zipped positionally with the request.
            by_id = {chunk.chunk_id: chunk for chunk in chunks}
            evidence: List[Evidence] = []
            for number, chunk_id in enumerate(chunk_ids, start=1):
                chunk = by_id.get(chunk_id)
                if chunk is not None:
                    evidence.append(self._map_chunk(number, chunk))
            return evidence
        except Exception as exc:
            raise map_engine_error(exc, context="Loading cited excerpts")

    # ---------- ingestion ----------

    def ingest(self, path: str, mode: IngestMode) -> IngestionSummary:
        engine = self._require_engine()
        try:
            ingest = {
                IngestMode.VIDEO: engine.ingest_videos,
                IngestMode.AUDIO: engine.ingest_audios,
                IngestMode.DOCUMENT: engine.ingest_documents,
                IngestMode.IMAGE: engine.ingest_images,
            }[mode]
            report = ingest(path)
            return IngestionSummary(
                target_path=report.target_path,
                scanned=report.total_files_scanned,
                succeeded=report.successful_ingestions,
                failed=report.failed_ingestions,
                vectors_created=report.vectors_created,
                clips_extracted=report.clips_extracted,
                pages_converted=report.pages_converted,
                images_described=report.images_described,
                duration_seconds=report.total_duration_seconds,
                errors=tuple(sorted(report.errors.items())),
            )
        except Exception as exc:
            raise map_engine_error(exc, context=f"{mode.value.capitalize()} ingestion")

    # ---------- mapping ----------

    def _map_qa_response(self, response: Any) -> ChatMessage:
        evidence = self._map_citations(response.citations)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            text=self._normalized_answer(response),
            outcome=TurnOutcome.ANSWERED,
            evidence=evidence,
            latency_ms=response.inference_time_ms,
            created_at=datetime.now(timezone.utc),
        )

    def _map_reasoned(self, result: Any) -> ChatMessage:
        reasoning = tuple(
            ReasoningStep(
                iteration=step.iteration,
                decision=step.decision,
                interim_analysis=step.interim_analysis,
                refined_query=step.refined_query,
                chunks_added=step.chunks_added,
            )
            for step in result.iteration_trace
        )

        if result.qa_response is None:
            # The reasoner wants a narrower question. This is the common branch, not an edge case.
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                text=result.clarification_question or "Could you narrow that question?",
                outcome=TurnOutcome.NEEDS_CLARIFICATION,
                reasoning=reasoning,
                total_iterations=result.total_iterations,
                created_at=datetime.now(timezone.utc),
            )

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            text=self._normalized_answer(result.qa_response),
            outcome=TurnOutcome.ANSWERED,
            evidence=self._map_citations(result.qa_response.citations),
            reasoning=reasoning,
            critique_unresolved=result.critique_unresolved,
            total_iterations=result.total_iterations,
            latency_ms=result.qa_response.inference_time_ms,
            created_at=datetime.now(timezone.utc),
        )

    def _normalized_answer(self, response: Any) -> str:
        """
        `QAResponse.generated_answer` is the raw model text, while `citations` was parsed from a
        normalized copy. So the text can still hold a grouped tag ("[Doc 1, 3]") that the citation
        list has already split into separate entries. Re-normalizing with the citation list's own
        numbers makes text and citations agree, which is what lets the bubble turn each `[Doc N]`
        into a link with a plain regex.
        """
        modules = _import_engine()
        valid_numbers = {
            number for number in (self._doc_number(c.doc_id) for c in response.citations)
            if number is not None
        }
        try:
            return modules["normalize_citation_tags"](
                response.generated_answer, 0, valid_numbers
            )
        except Exception as exc:
            logger.warning(f"Citation normalization failed, showing raw answer text: {exc}")
            return response.generated_answer

    def _map_citations(self, citations: Sequence[Any]) -> Tuple[Evidence, ...]:
        """
        Citations carry ids, timestamps and a resolved path but no text, so the excerpts are
        fetched in one extra indexed read (no model, no VRAM). If that read fails the cards still
        render with their timespan and clip -- an answer must never be shown without its evidence
        just because the excerpt lookup had a bad day.
        """
        excerpts: Dict[uuid.UUID, Any] = {}
        chunk_ids = [c.chunk_id for c in citations]
        if chunk_ids and self._engine is not None:
            try:
                excerpts = {
                    chunk.chunk_id: chunk for chunk in self._engine.get_chunks(chunk_ids)
                }
            except Exception as exc:
                logger.warning(f"Could not hydrate citation excerpts: {exc}")

        evidence: List[Evidence] = []
        for citation in citations:
            chunk = excerpts.get(citation.chunk_id)
            number = self._doc_number(citation.doc_id)
            evidence.append(Evidence(
                doc_number=number if number is not None else len(evidence) + 1,
                chunk_id=citation.chunk_id,
                source_id=citation.source_id,
                start_sec=citation.start_sec,
                end_sec=citation.end_sec,
                excerpt=getattr(chunk, "raw_text", "") or "",
                speakers=tuple(getattr(chunk, "speaker_labels", ()) or ()),
                clip_path=citation.physical_path,
                media_kind=self._media_kind(citation.physical_path, citation.extracted_file_id,
                                            self._is_document(citation), self._is_image(citation)),
                location=self._location(citation),
            ))
        return tuple(evidence)

    def _map_hit(self, rank: int, chunk: Any) -> SearchHit:
        path = getattr(chunk, "physical_path", None)
        return SearchHit(
            rank=rank,
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            # raw_text is the verbatim transcript; text_payload carries generated context that is
            # not citable evidence. Fall back only when a chunk has no separate raw text.
            text=chunk.raw_text or chunk.text_payload,
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            relevance_score=chunk.relevance_score,
            rerank_score=chunk.rerank_score,
            speakers=tuple(chunk.speaker_labels or ()),
            clip_path=path,
            media_kind=self._media_kind(path, chunk.extracted_file_id, self._is_document(chunk),
                                        self._is_image(chunk)),
            location=self._location(chunk),
        )

    def _map_chunk(self, number: int, chunk: Any) -> Evidence:
        path = getattr(chunk, "physical_path", None)
        return Evidence(
            doc_number=number,
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            excerpt=chunk.raw_text or chunk.text_payload,
            speakers=tuple(chunk.speaker_labels or ()),
            clip_path=path,
            media_kind=self._media_kind(path, chunk.extracted_file_id, self._is_document(chunk),
                                        self._is_image(chunk)),
            location=self._location(chunk),
            # A hydration read is unranked: relevance is hard-coded 0.0 and rerank is None
            # upstream, so leaving both unset keeps meaningless numbers off the card.
        )

    @staticmethod
    def _media_kind(path: Optional[str], extracted_file_id: Optional[uuid.UUID],
                    is_document: bool = False, is_image: bool = False) -> MediaKind:
        """
        A chunk with an extracted clip gets a file that *is* the excerpt. Without one, the engine
        falls back to the entire original source (the audio-only pipeline cuts no clips), which
        the player has to seek into. A document opens whole; its citation names the page. An
        image opens whole too, and its card previews it.
        """
        if not path:
            return MediaKind.UNRESOLVED
        if is_image:
            return MediaKind.IMAGE
        if is_document:
            return MediaKind.DOCUMENT
        return MediaKind.CLIP if extracted_file_id else MediaKind.FULL_SOURCE

    @staticmethod
    def _anchor_value(item: Any) -> Optional[str]:
        kind = getattr(item, "anchor_kind", None)
        return getattr(kind, "value", kind)

    @classmethod
    def _is_document(cls, item: Any) -> bool:
        """
        A chunk, citation or report entry anchored by anything but time: page, lines, section or
        image. A missing anchor is a recording, as it was before documents existed.
        """
        return cls._anchor_value(item) not in (None, "time")

    @classmethod
    def _is_image(cls, item: Any) -> bool:
        return cls._anchor_value(item) == "image"

    def _location(self, item: Any) -> str:
        """A document excerpt's "p. 7 · Article 7" label; empty for a recording, shown by its time range."""
        if not self._is_document(item):
            return ""
        kind = getattr(item.anchor_kind, "value", item.anchor_kind)
        return _import_engine()["format_location"](
            kind, page_start=item.page_start, page_end=item.page_end, section_path=item.section_path)

    @staticmethod
    def _doc_number(doc_id: str) -> Optional[int]:
        digits = "".join(ch for ch in str(doc_id) if ch.isdigit())
        return int(digits) if digits else None

    def _on_job_progress(self, phase: str, state: str, source_id, detail=None) -> None:
        """
        Called by JobManager on the worker thread, once per file per phase. It must stay trivial:
        the listener above it only emits a Qt signal.
        """
        if self._progress is None:
            return
        try:
            self._progress(ProgressTick(
                phase=str(phase),
                state=JobProgressState(state),
                source_id=source_id,
                detail=detail,
            ))
        except Exception as exc:
            logger.debug(f"Progress listener rejected a tick: {exc}")

    def _require_engine(self):
        if self._engine is None:
            raise ApplianceOfflineError("The appliance is not running. Start it first.")
        return self._engine
