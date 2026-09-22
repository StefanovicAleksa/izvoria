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
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

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

_PHASES = [
    "scan", "ingest", "audio_extraction", "vad", "transcribe",
    "diarize", "contextual_retrieval", "context_unification", "rag_indexing",
]


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

    # ---------- lifecycle ----------

    def boot(self) -> None:
        time.sleep(min(self._latency, 2.0))
        self._booted = True

    def shutdown(self) -> None:
        self._booted = False

    def is_ingesting(self) -> bool:
        return self._ingesting

    def set_progress_listener(self, listener: Optional[Callable[..., None]]) -> None:
        self._progress = listener

    # ---------- live inference ----------

    def chat(self, prompt: str, session_id: uuid.UUID, mode: ChatMode, limit: int) -> ChatMessage:
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
        return SearchOutcome(
            query=prompt,
            hits=tuple(hits),
            execution_time_ms=int(self._latency * 500),
        )

    def hydrate_evidence(self, chunk_ids: List[uuid.UUID]) -> List[Evidence]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    # ---------- ingestion ----------

    def ingest(self, path: str, mode: IngestMode) -> IngestionSummary:
        self._ingesting = True
        try:
            files = 3
            for phase in _PHASES:
                for index in range(files):
                    self._tick(phase, JobProgressState.STARTED, index)
                    time.sleep(self._latency * 0.12)
                    self._tick(phase, JobProgressState.COMPLETED, index)
            return IngestionSummary(
                target_path=path,
                scanned=files,
                succeeded=files - 1,
                failed=1,
                vectors_created=184,
                clips_extracted=57,
                duration_seconds=self._latency * len(_PHASES) * files,
                errors=(("rag_indexing_error_0", "Stub: one file failed on purpose"),),
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
        return ReportPlanDraft(
            report_id=self._report_id,
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

    def load_report(self, report_id: uuid.UUID) -> Optional[ReportDocument]:
        return self._last_report

    def _report(self, plan: ReportPlanDraft, mode: ReportMode, status: ReportRunStatus,
                sections: int, uncovered: int) -> ReportDocument:
        evidence = self._evidence(3)
        body = [f"# {plan.topic}", ""]
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
        return document

    def _tick(self, phase: str, state: JobProgressState, index: int) -> None:
        if self._progress is None:
            return
        self._progress(ProgressTick(
            phase=phase,
            state=state,
            source_id=uuid.uuid4(),
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
