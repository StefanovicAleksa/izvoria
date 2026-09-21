# Canvas: desktop_interfaces_v1
# Path: frontend/onyx_desktop/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from onyx_desktop.domain.entities import (
    ChatMessage,
    Evidence,
    IngestionSummary,
    SearchOutcome,
)
from onyx_desktop.domain.enums import ChatMode, IngestMode, SearchMode


class IEngineGateway(ABC):
    """
    The port between the UI and the Onyx engine. Everything above this line is Qt and plain
    dataclasses; everything below it is the appliance.

    Implementations must be callable from a single background worker thread only. They are not
    required to be thread-safe, because every call is serialized onto one thread by design: the
    VRAM pool holds one model at a time, so concurrent calls would collide rather than queue.

    Implementations raise only the desktop exception hierarchy
    (frontend/onyx_desktop/exceptions/), never engine or third-party exceptions.
    """

    @abstractmethod
    def boot(self) -> None:
        """Builds and starts the engine. Slow: DB DDL, GC, model manager construction."""

    @abstractmethod
    def shutdown(self) -> None:
        """Releases VRAM, stops background model workers, disposes DB pools."""

    @abstractmethod
    def is_ingesting(self) -> bool:
        """True while a batch job holds the hardware lock, so the UI can refuse live calls."""

    @abstractmethod
    def chat(
        self,
        prompt: str,
        session_id: uuid.UUID,
        mode: ChatMode,
        limit: int,
    ) -> ChatMessage:
        """
        One conversation turn. Returns an assistant message that is either an answer with its
        evidence, or -- for Deep Research -- a clarification question back at the user.
        """

    @abstractmethod
    def search(self, prompt: str, mode: SearchMode, limit: int) -> SearchOutcome:
        """Ranked retrieval, with or without the SLM relevance scrubber."""

    @abstractmethod
    def ingest(self, path: str, mode: IngestMode) -> IngestionSummary:
        """
        Runs a batch pipeline to completion. Very long and not interruptible; blocks live
        inference for its whole duration.
        """

    @abstractmethod
    def hydrate_evidence(self, chunk_ids: List[uuid.UUID]) -> List[Evidence]:
        """
        Fetches verbatim excerpts and clip paths for chunk ids. Needed because a citation
        carries ids and timestamps but no text. Cheap: an indexed read, no model, no VRAM.
        """

    def set_progress_listener(self, listener: Optional[Callable[..., None]]) -> None:
        """
        Registers a per-file pipeline progress callback. Optional: a gateway that cannot report
        progress may ignore it, so the UI degrades to its log pane instead of breaking.
        """
        return None
