# Canvas: desktop_stub_gateway_v1
# Path: frontend/onyx_desktop/data/stub_gateway.py

"""
A gateway with no engine behind it.

Booting the real appliance takes a long time, needs Postgres up, model weights on disk and a free
GPU, and a single chat turn costs tens of seconds. That is a hopeless loop for iterating on a
window. This stub answers instantly-ish with canned data covering every branch the UI has to
render: a grounded answer, an answer with *no* evidence, a Reasoned Answer clarification request,
an unresolved clip path, and a failure.
"""

import itertools
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from onyx_desktop.domain.entities import (
    ChatMessage,
    ConversationRef,
    Evidence,
    IngestionSummary,
    LanguageOption,
    ProgressTick,
    ReasoningStep,
    ReportDocument,
    ReportOutlineDraft,
    ReportPlanDraft,
    ReportRef,
    SearchHit,
    SearchOutcome,
    SearchRecordRef,
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
from onyx_desktop.exceptions.gateway import EngineCallFailedError, InvalidRequestError

_LOREM = (
    "SPEAKER_01: The retention schedule was circulated on the fourteenth, and both parties "
    "acknowledged receipt in writing.\n"
    "SPEAKER_02: That matches our copy of the cover letter."
)

# The report phases the engine emits, so the stub's progress bar exercises the same routing.
_REPORT_PHASES = [
    "research_scoping", "rag_retrieval", "rag_reranking",
    "research_outline", "research_section", "research_section", "research_summary",
]

_LANGUAGES = (
    LanguageOption(code="en", name="English"),
    LanguageOption(code="sr", name="Srpski"),
)

# The jobs each batch pipeline reports, in order (app/orchestration/service/pipelines/), so the Ingest view's
# steps move the way they do on the real appliance.
_AUDIO_JOBS = ["vad", "domain_adaptation", "transcribe", "hallucination_correction", "diarize",
               "sentence_reconstruction", "phonetic_correction", "contextual_retrieval"]
_PHASES_BY_MODE = {
    IngestMode.VIDEO: ["file_ingest", "audio_extraction", *_AUDIO_JOBS, "visual_cue_detection",
                       "frame_extraction", "visual_analysis", "visual_hallucination_correction",
                       "visual_context_synthesis", "context_unification", "rag_indexing", "clip_extraction"],
    IngestMode.AUDIO: ["file_ingest", "audio_extraction", *_AUDIO_JOBS, "context_unification", "rag_indexing"],
    IngestMode.DOCUMENT: ["file_ingest", "document_conversion", "document_contextualization",
                          "context_unification", "rag_indexing"],
    IngestMode.IMAGE: ["file_ingest", "image_description", "context_unification", "rag_indexing"],
}


_NEW = object()  # a fresh source id per tick, for the report phases that have no file behind them


class StubEngineGateway(IEngineGateway):
    """Canned responses with believable latency. Never touches app.*, the GPU or the database."""

    def __init__(self, latency_seconds: float = 1.2, clip_path: Optional[str] = None):
        self._latency = latency_seconds
        self._clip_path = clip_path
        self._booted = False
        self._ingesting = False
        self._turn_counter = itertools.count(1)
        self._progress: Optional[Callable[[ProgressTick], None]] = None
        self._chunks = {}
        self._report_counter = itertools.count(1)
        self._cancel_requested = False
        self._last_report: Optional[ReportDocument] = None
        self._report_id: Optional[uuid.UUID] = None
        self._language = "en"
        # What each batch was ingested in, so a test can see the language reached the gateway
        self.ingested_languages: List[str] = []

        # History, as the database would hold it. Read from the bridge's reader thread while the
        # worker thread writes, hence the lock.
        self._history_lock = threading.Lock()
        self._conversations: Dict[uuid.UUID, dict] = {}
        self._searches: Dict[uuid.UUID, Tuple[SearchRecordRef, SearchOutcome]] = {}
        self._reports: Dict[uuid.UUID, dict] = {}
        self._seed_history()

    # ---------- lifecycle ----------

    def boot(self) -> None:
        time.sleep(min(self._latency, 2.0))
        self._booted = True

    def shutdown(self) -> None:
        self._booted = False

    def is_ingesting(self) -> bool:
        return self._ingesting

    # ---------- language ----------

    def languages(self) -> List[LanguageOption]:
        return list(_LANGUAGES)

    def active_language(self) -> str:
        return self._language

    def switch_language(self, code: str) -> None:
        if code not in {option.code for option in _LANGUAGES}:
            raise InvalidRequestError(f"'{code}' is not a supported language.")
        if code != self._language:
            time.sleep(min(self._latency, 1.0))
            self._language = code

    def set_progress_listener(self, listener: Optional[Callable[..., None]]) -> None:
        self._progress = listener

    # ---------- live inference ----------

    def chat(self, prompt: str, session_id: uuid.UUID, mode: ChatMode, limit: int) -> ChatMessage:
        message = self._answer(prompt, mode, limit)
        if message.outcome is TurnOutcome.ANSWERED:
            self._remember_turn(session_id, prompt, message)
        return message

    def _answer(self, prompt: str, mode: ChatMode, limit: int) -> ChatMessage:
        if not prompt.strip():
            raise InvalidRequestError("A question cannot be empty.")
        time.sleep(self._latency)
        turn = next(self._turn_counter)

        # Cycle the branches so every render path gets exercised while clicking around.
        if "fail" in prompt.lower():
            raise EngineCallFailedError("Stub failure, on request.")

        if mode is ChatMode.REASONING and turn % 2 == 1:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                text="Which retention schedule do you mean — the 2024 revision, or the "
                     "original circulated in 2019?",
                outcome=TurnOutcome.NEEDS_CLARIFICATION,
                reasoning=self._reasoning(),
                total_iterations=2,
                latency_ms=int(self._latency * 1000),
                created_at=datetime.now(timezone.utc),
            )

        ungrounded = turn % 4 == 0
        evidence = () if ungrounded else self._evidence(limit)
        text = (
            "I could not find anything in the ingested material that answers that."
            if ungrounded
            else "The schedule was circulated on the fourteenth [Doc 1] and receipt was "
                 "acknowledged in writing by both parties [Doc 2]."
        )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            text=text,
            outcome=TurnOutcome.ANSWERED,
            evidence=evidence,
            reasoning=self._reasoning() if mode is ChatMode.REASONING else (),
            critique_unresolved=(mode is ChatMode.REASONING and turn % 3 == 0),
            total_iterations=2 if mode is ChatMode.REASONING else 0,
            latency_ms=int(self._latency * 1000),
            created_at=datetime.now(timezone.utc),
        )

    def search(self, prompt: str, mode: SearchMode, limit: int) -> SearchOutcome:
        if not prompt.strip():
            raise InvalidRequestError("A search query cannot be empty.")
        time.sleep(self._latency * 0.5)
        hits = []
        for rank in range(1, min(limit, 5) + 1):
            chunk_id = uuid.uuid4()
            hits.append(SearchHit(
                rank=rank,
                chunk_id=chunk_id,
                source_id=uuid.uuid4(),
                text=f"{_LOREM}\n(stub result {rank} for {mode.value} search)",
                start_sec=60.0 * rank,
                end_sec=60.0 * rank + 36.0,
                relevance_score=0.9 - rank * 0.07,
                rerank_score=None if mode is SearchMode.STANDARD else 0.8 - rank * 0.05,
                speakers=("SPEAKER_01", "SPEAKER_02"),
                clip_path=self._clip_path,
                media_kind=MediaKind.CLIP if self._clip_path else MediaKind.UNRESOLVED,
            ))
        outcome = SearchOutcome(
            query=prompt,
            hits=tuple(hits),
            execution_time_ms=int(self._latency * 500),
            mode=mode,
        )
        self._remember_search(outcome)
        return outcome

    def hydrate_evidence(self, chunk_ids: List[uuid.UUID]) -> List[Evidence]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    # ---------- ingestion ----------

    def ingest(self, path: str, mode: IngestMode, language: str) -> IngestionSummary:
        self.switch_language(language)
        self.ingested_languages.append(language)
        self._ingesting = True
        try:
            files = 3
            sources = [uuid.uuid4() for _ in range(files)]
            phases = _PHASES_BY_MODE[mode]
            self._tick("directory_scan", JobProgressState.COMPLETED, 0, source_id=None)
            for phase in phases:
                for index in range(files):
                    self._tick(phase, JobProgressState.STARTED, index, source_id=sources[index])
                    time.sleep(self._latency * 0.05)
                    self._tick(phase, JobProgressState.COMPLETED, index, source_id=sources[index])
            return IngestionSummary(
                target_path=path,
                scanned=files,
                succeeded=files - 1,
                failed=1,
                vectors_created=184,
                clips_extracted=0 if mode in (IngestMode.DOCUMENT, IngestMode.IMAGE) else 57,
                pages_converted=42 if mode is IngestMode.DOCUMENT else 0,
                images_described=files - 1 if mode is IngestMode.IMAGE else 0,
                duration_seconds=self._latency * len(phases) * files,
                errors=((f"{path.rstrip('/')}/stub_file_2.mp4",
                         "Whisper heard 'de' (probability 0.97) in source stub, not English, so it was not transcribed"),),
            )
        finally:
            self._ingesting = False

    # ---------- helpers ----------

    # ==========================================
    # RESEARCH REPORTS
    # ==========================================

    def propose_report_plan(self, topic: str, mode: ReportMode) -> ReportPlanDraft:
        time.sleep(self._latency)
        if "fail" in topic.lower():
            raise EngineCallFailedError("Stub failure, on request.")

        self._report_id = uuid.uuid4()
        plan = self._plan(self._report_id, topic)
        self._remember_report(ReportRef(report_id=plan.report_id, topic=topic,
                                        status=ReportRunStatus.AWAITING_APPROVAL,
                                        created_at=datetime.now(timezone.utc)), plan=plan)
        return plan

    @staticmethod
    def _plan(report_id: uuid.UUID, topic: str) -> ReportPlanDraft:
        return ReportPlanDraft(
            report_id=report_id,
            topic=topic,
            scope_note=f"A stub plan for {topic!r}. Nothing was searched.",
            sections=(
                ReportOutlineDraft(key="s1", title="Background", brief="What the archive establishes.",
                                   questions=("What is the background?",)),
                ReportOutlineDraft(key="s2", title="Key findings", brief="The substantive findings.",
                                   questions=("What did they conclude?", "Who said it?")),
                ReportOutlineDraft(key="s3", title="Open questions", brief="What is unresolved.",
                                   questions=("What remains unanswered?",)),
            ),
            indexed_source_count=3,
            indexed_chunk_count=91,
            expected_job_count=10,
            estimated_minutes=12,
        )

    def run_report(self, plan: ReportPlanDraft, mode: ReportMode) -> ReportDocument:
        # Progress ticks first, so the determinate bar has something to move against.
        for index, phase in enumerate(_REPORT_PHASES):
            self._tick(phase, JobProgressState.STARTED, index)
            time.sleep(self._latency / 3)
            self._tick(phase, JobProgressState.COMPLETED, index)

        if self._cancel_requested:
            self._cancel_requested = False
            return self._report(plan, mode, ReportRunStatus.CANCELLED, sections=1, uncovered=0)

        # Cycles the render branches: a whole report, then one with an uncovered section.
        run = next(self._report_counter)
        if run % 2 == 0:
            return self._report(plan, mode, ReportRunStatus.COMPLETED, sections=2, uncovered=1)
        return self._report(plan, mode, ReportRunStatus.COMPLETED, sections=3, uncovered=0)

    def request_report_cancel(self, report_id: uuid.UUID) -> bool:
        """Records the request; the next run_report returns a partial report, as the engine would."""
        self._cancel_requested = True
        return True

    def resume_report(self, report_id: uuid.UUID) -> ReportDocument:
        with self._history_lock:
            entry = self._reports.get(report_id)
        if entry is None or not entry["ref"].resumable:
            raise InvalidRequestError("Only a stopped report with gathered evidence can be resumed.")
        for index, phase in enumerate(("research_section", "research_summary")):
            self._tick(phase, JobProgressState.STARTED, index)
            time.sleep(self._latency / 3)
            self._tick(phase, JobProgressState.COMPLETED, index)
        return self._report(entry["plan"], ReportMode.DEEP, ReportRunStatus.COMPLETED, sections=3, uncovered=0)

    # ---------- history ----------

    def list_conversations(self) -> List[ConversationRef]:
        with self._history_lock:
            refs = [c["ref"] for c in self._conversations.values()]
        return sorted(refs, key=lambda r: r.last_at, reverse=True)

    def load_conversation(self, session_id: uuid.UUID) -> List[ChatMessage]:
        with self._history_lock:
            entry = self._conversations.get(session_id)
            if entry is None:
                raise InvalidRequestError("That conversation is no longer in the history.")
            return list(entry["messages"])

    def delete_conversation(self, session_id: uuid.UUID) -> None:
        with self._history_lock:
            self._conversations.pop(session_id, None)

    def list_searches(self) -> List[SearchRecordRef]:
        with self._history_lock:
            refs = [ref for ref, _ in self._searches.values()]
        return sorted(refs, key=lambda r: r.created_at, reverse=True)

    def load_search(self, record_id: uuid.UUID) -> SearchOutcome:
        with self._history_lock:
            entry = self._searches.get(record_id)
        if entry is None:
            raise InvalidRequestError("That search is no longer in the history.")
        ref, outcome = entry
        return replace(outcome, saved_at=ref.created_at)

    def delete_search(self, record_id: uuid.UUID) -> None:
        with self._history_lock:
            self._searches.pop(record_id, None)

    def list_reports(self) -> List[ReportRef]:
        with self._history_lock:
            refs = [r["ref"] for r in self._reports.values()]
        epoch = datetime.min.replace(tzinfo=timezone.utc)
        return sorted(refs, key=lambda r: r.created_at or epoch, reverse=True)

    def load_report(self, report_id: uuid.UUID) -> ReportDocument:
        with self._history_lock:
            entry = self._reports.get(report_id)
        if entry is None or entry.get("document") is None:
            raise InvalidRequestError("That report has nothing written yet.")
        return entry["document"]

    def load_report_plan(self, report_id: uuid.UUID) -> ReportPlanDraft:
        with self._history_lock:
            entry = self._reports.get(report_id)
        if entry is None:
            raise InvalidRequestError("That research plan is no longer available; scope it again.")
        return entry["plan"]

    def _remember_turn(self, session_id: uuid.UUID, question: str, answer: ChatMessage) -> None:
        now = datetime.now(timezone.utc)
        with self._history_lock:
            entry = self._conversations.setdefault(session_id, {"messages": [], "started": now, "title": question})
            entry["messages"] += [ChatMessage(role=MessageRole.USER, text=question, created_at=now), answer]
            entry["ref"] = ConversationRef(session_id=session_id, title=entry["title"], started_at=entry["started"],
                                           last_at=now, turn_count=len(entry["messages"]) // 2)

    def _remember_search(self, outcome: SearchOutcome) -> None:
        ref = SearchRecordRef(record_id=uuid.uuid4(), query=outcome.query, mode=outcome.mode or SearchMode.STANDARD,
                              hit_count=len(outcome.hits), created_at=datetime.now(timezone.utc))
        with self._history_lock:
            self._searches[ref.record_id] = (ref, outcome)

    def _remember_report(self, ref: ReportRef, plan: Optional[ReportPlanDraft] = None,
                         document: Optional[ReportDocument] = None) -> None:
        with self._history_lock:
            entry = self._reports.setdefault(ref.report_id, {})
            entry["ref"] = ref
            if plan is not None:
                entry["plan"] = plan
            if document is not None:
                entry["document"] = document

    def _seed_history(self) -> None:
        """One of everything the history panes have to render, so --stub shows them on first launch."""
        earlier = datetime.now(timezone.utc) - timedelta(days=2)
        session_id = uuid.uuid4()
        self._remember_turn(session_id, "When was the retention schedule circulated?", ChatMessage(
            role=MessageRole.ASSISTANT,
            text="It was circulated on the fourteenth [Doc 1], and both parties acknowledged it [Doc 2].",
            evidence=self._evidence(2), created_at=earlier))
        self._remember_turn(session_id, "Who acknowledged it first?", ChatMessage(
            role=MessageRole.ASSISTANT, text="The record does not say who acknowledged it first.",
            missing_evidence=1, created_at=earlier))
        self._conversations[session_id]["ref"] = replace(
            self._conversations[session_id]["ref"], started_at=earlier, last_at=earlier)

        self._remember_search(SearchOutcome(
            query="retention schedule cover letter", mode=SearchMode.VERIFIED, execution_time_ms=640,
            hits=tuple(SearchHit(rank=r, chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text=_LOREM,
                                 start_sec=60.0 * r, end_sec=60.0 * r + 36.0, relevance_score=0.9 - r * 0.1,
                                 speakers=("SPEAKER_01",)) for r in (1, 2)),
            missing_hits=1))

        for topic, status, evidence in (
                ("Retention obligations across the archive", ReportRunStatus.COMPLETED, 3),
                ("Who approved the 2024 revision", ReportRunStatus.AWAITING_APPROVAL, 0),
                ("Correspondence about the audit", ReportRunStatus.CANCELLED, 4),
        ):
            plan = self._plan(uuid.uuid4(), topic)
            document = None
            if status is not ReportRunStatus.AWAITING_APPROVAL:
                sections = 3 if status is ReportRunStatus.COMPLETED else 1
                document = self._report(plan, ReportMode.DEEP, status, sections=sections, uncovered=0)
            self._remember_report(ReportRef(report_id=plan.report_id, topic=topic, status=status,
                                            created_at=earlier, section_count=3, evidence_count=evidence),
                                  plan=plan, document=document)

    def _report(self, plan: ReportPlanDraft, mode: ReportMode, status: ReportRunStatus,
                sections: int, uncovered: int) -> ReportDocument:
        evidence = self._evidence(3)
        body = [f"# {plan.title or plan.topic}", ""]
        body.append(
            f"> Generated by Onyx (stub) from {len(evidence)} cited excerpts across 1 source.\n"
            f"> Coverage: swept 3 of 3 indexed sources.\n"
            f"> Searched in {'2 rounds' if mode is ReportMode.EXHAUSTIVE else 'a single round'} "
            f"({mode.value} mode).\n"
            f"> Retrieval scrubber: disabled."
        )
        body.append("")
        for index, section in enumerate(plan.sections[:sections]):
            body += [f"## {section.title}", "",
                     f"{_LOREM} [Doc {index + 1}](onyx://doc/{index + 1})", ""]
        if uncovered:
            body += ["## Not covered by the available material", "",
                     f"- **{plan.sections[-1].title}** — no evidence in the indexed corpus supported this.",
                     ""]
        body += ["## References", "", "| Doc | Source | Timespan | Speakers |", "| --- | --- | --- | --- |"]
        for item in evidence[:sections]:
            body.append(f"| [Doc {item.doc_number}] | Stub recording | 00:00:10–00:00:40 | speaker_0 |")

        document = ReportDocument(
            report_id=plan.report_id,
            topic=plan.topic,
            title=plan.title,
            markdown="\n".join(body) + "\n",
            status=status,
            mode=mode,
            evidence=evidence[:sections],
            markdown_path=str(Path(tempfile.gettempdir()) / f"{plan.report_id}.md"),
            section_count=sections,
            uncovered_count=uncovered,
            rounds_used=2 if mode is ReportMode.EXHAUSTIVE else 1,
            indexed_source_count=3,
            swept_source_count=3,
        )
        self._last_report = document
        self._remember_report(ReportRef(
            report_id=plan.report_id, topic=plan.topic, status=status, created_at=datetime.now(timezone.utc),
            section_count=len(plan.sections), evidence_count=len(evidence), title=plan.title),
            plan=plan, document=document)
        return document

    def _tick(self, phase: str, state: JobProgressState, index: int, source_id=_NEW) -> None:
        if self._progress is None:
            return
        self._progress(ProgressTick(
            phase=phase,
            state=state,
            source_id=uuid.uuid4() if source_id is _NEW else source_id,
            detail=f"stub_file_{index}.mp4",
        ))

    def _evidence(self, limit: int) -> tuple:
        items = []
        for number in range(1, min(limit, 2) + 1):
            chunk_id = uuid.uuid4()
            # The second card deliberately has no clip, to exercise the disabled-player path.
            resolved = self._clip_path if number == 1 else None
            evidence = Evidence(
                doc_number=number,
                chunk_id=chunk_id,
                source_id=uuid.uuid4(),
                start_sec=72.0 * number,
                end_sec=72.0 * number + 36.0,
                excerpt=_LOREM,
                speakers=("SPEAKER_01", "SPEAKER_02"),
                clip_path=resolved,
                media_kind=MediaKind.CLIP if resolved else MediaKind.UNRESOLVED,
                anchor_kind="time",
            )
            self._chunks[chunk_id] = evidence
            items.append(evidence)
        return tuple(items)

    @staticmethod
    def _reasoning() -> tuple:
        return (
            ReasoningStep(
                iteration=1,
                decision="SEARCH",
                interim_analysis="No evidence yet about the circulation date; searching for the "
                                 "retention schedule and its cover letter.",
                refined_query="retention schedule circulation date acknowledgement",
                chunks_added=4,
            ),
            ReasoningStep(
                iteration=2,
                decision="ANSWER",
                interim_analysis="Two chunks state the date and the written acknowledgement; "
                                 "that is enough to answer without guessing.",
                chunks_added=0,
            ),
        )
