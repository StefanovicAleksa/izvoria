# Canvas: desktop_entities_v1
# Path: frontend/onyx_desktop/domain/entities.py

"""
The UI's own value objects. Deliberately frozen stdlib dataclasses rather than the engine's
pydantic entities: these cross a thread boundary into Qt slots, and keeping them free of any
`app.*` import is what lets the whole UI run against a stub gateway with no GPU, DB or weights.

The gateway (frontend/onyx_desktop/data/engine_gateway.py) is the only place engine entities are
translated into these.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from onyx_desktop.domain.enums import (
    ApplianceStatus,
    JobProgressState,
    MediaKind,
    MessageRole,
    TurnOutcome,
)


@dataclass(frozen=True)
class Evidence:
    """
    One cited excerpt, ready to render and play.

    `excerpt` is the engine's verbatim speaker-labelled transcript text (`RetrievedChunk.raw_text`),
    never `text_payload` -- the latter carries model-generated situating context that is explicitly
    not citable source material.
    """
    doc_number: int
    chunk_id: uuid.UUID
    source_id: uuid.UUID
    start_sec: float
    end_sec: float
    excerpt: str = ""
    speakers: Tuple[str, ...] = ()
    clip_path: Optional[str] = None
    media_kind: MediaKind = MediaKind.UNRESOLVED

    # Only meaningful for ranked search hits; a hydration read returns 0.0/None for these,
    # so they stay out of citation cards.
    relevance_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @property
    def doc_label(self) -> str:
        return f"[Doc {self.doc_number}]"

    @property
    def is_playable(self) -> bool:
        return self.clip_path is not None and self.media_kind is not MediaKind.UNRESOLVED

    @property
    def seek_to_sec(self) -> float:
        """A per-chunk clip already *is* the excerpt; a full source has to be seeked."""
        return 0.0 if self.media_kind is MediaKind.CLIP else self.start_sec


@dataclass(frozen=True)
class ReasoningStep:
    """One iteration of the Deep Research reasoner loop, for the collapsible trace panel."""
    iteration: int
    decision: str
    interim_analysis: str
    refined_query: Optional[str] = None
    chunks_added: int = 0


@dataclass(frozen=True)
class ChatMessage:
    """
    A rendered conversation turn.

    `text` is already citation-normalized by the gateway, so a plain `[Doc N]` scan is safe for
    building inline links (the engine returns the raw model text, which can still hold grouped
    tags like `[Doc 1, 3]` that the citation list has already split apart).
    """
    role: MessageRole
    text: str
    outcome: TurnOutcome = TurnOutcome.ANSWERED
    evidence: Tuple[Evidence, ...] = ()
    reasoning: Tuple[ReasoningStep, ...] = ()
    critique_unresolved: bool = False
    total_iterations: int = 0
    latency_ms: int = 0
    created_at: Optional[datetime] = None

    @property
    def is_ungrounded(self) -> bool:
        """
        An answer with no evidence is the failure mode this product exists to prevent, so the UI
        flags it rather than rendering it as an ordinary reply.
        """
        return (
            self.role is MessageRole.ASSISTANT
            and self.outcome is TurnOutcome.ANSWERED
            and not self.evidence
        )


@dataclass(frozen=True)
class SearchHit:
    """One ranked retrieval result for the Search view."""
    rank: int
    chunk_id: uuid.UUID
    source_id: uuid.UUID
    text: str
    start_sec: float
    end_sec: float
    relevance_score: float
    rerank_score: Optional[float] = None
    speakers: Tuple[str, ...] = ()
    clip_path: Optional[str] = None
    media_kind: MediaKind = MediaKind.UNRESOLVED

    def as_evidence(self) -> Evidence:
        """Search hits feed the same evidence card and player as citations do."""
        return Evidence(
            doc_number=self.rank,
            chunk_id=self.chunk_id,
            source_id=self.source_id,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            excerpt=self.text,
            speakers=self.speakers,
            clip_path=self.clip_path,
            media_kind=self.media_kind,
            relevance_score=self.relevance_score,
            rerank_score=self.rerank_score,
        )


@dataclass(frozen=True)
class SearchOutcome:
    query: str
    hits: Tuple[SearchHit, ...]
    execution_time_ms: int = 0


@dataclass(frozen=True)
class IngestionSummary:
    target_path: str
    scanned: int = 0
    succeeded: int = 0
    failed: int = 0
    vectors_created: int = 0
    clips_extracted: int = 0
    duration_seconds: float = 0.0
    errors: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ProgressTick:
    """
    One unit of pipeline work, reported by the JobManager callback. The pipeline runs phase by
    phase over all files (not file by file through all phases), so progress is two-level:
    which phase, and how far through that phase's files.
    """
    phase: str
    state: JobProgressState
    source_id: Optional[uuid.UUID] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class LogLine:
    level: str
    logger_name: str
    message: str


@dataclass(frozen=True)
class SessionRef:
    """A conversation in the GUI-local registry. The engine only ever receives the id."""
    session_id: uuid.UUID
    title: str
    started_at: datetime
    turn_count: int = 0


@dataclass(frozen=True)
class StatusSnapshot:
    status: ApplianceStatus
    detail: str = ""
