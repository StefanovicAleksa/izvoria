# Canvas: desktop_stub_gateway_v1
# Path: frontend/onyx_desktop/data/stub_gateway.py

"""
A gateway with no engine behind it.

Booting the real appliance takes a long time, needs Postgres up, model weights on disk and a free
GPU, and a single chat turn costs tens of seconds. That is a hopeless loop for iterating on a
window. This stub answers instantly-ish with canned data covering every branch the UI has to
render: a grounded answer, an answer with *no* evidence, a Deep Research clarification request,
an unresolved clip path, and a failure.
"""

import itertools
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, List, Optional

from onyx_desktop.domain.entities import (
    ChatMessage,
    Evidence,
    IngestionSummary,
    ProgressTick,
    ReasoningStep,
    SearchHit,
    SearchOutcome,
)
from onyx_desktop.domain.enums import (
    ChatMode,
    IngestMode,
    JobProgressState,
    MediaKind,
    MessageRole,
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

        if mode is ChatMode.DEEP_RESEARCH and turn % 2 == 1:
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
            reasoning=self._reasoning() if mode is ChatMode.DEEP_RESEARCH else (),
            critique_unresolved=(mode is ChatMode.DEEP_RESEARCH and turn % 3 == 0),
            total_iterations=2 if mode is ChatMode.DEEP_RESEARCH else 0,
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
