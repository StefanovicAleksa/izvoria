# Canvas: desktop_interfaces_v1
# Path: frontend/izvoria_desktop/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from izvoria_desktop.domain.entities import (
    ChatMessage,
    ConversationRef,
    Evidence,
    IngestionSummary,
    LanguageOption,
    ReportDocument,
    ReportPlanDraft,
    ReportRef,
    SearchOutcome,
    SearchRecordRef,
)
from izvoria_desktop.domain.enums import ChatMode, IngestMode, ReportMode, SearchMode


class IEngineGateway(ABC):
    """
    The port between the UI and the Izvoria engine. Everything above this line is Qt and plain
    dataclasses; everything below it is the appliance.

    Implementations must be callable from a single background worker thread only. They are not
    required to be thread-safe, because every call is serialized onto one thread by design: the
    VRAM pool holds one model at a time, so concurrent calls would collide rather than queue.
    Two exceptions: request_report_cancel (GUI thread) and the history methods (reader thread),
    neither of which touches a model.

    Implementations raise only the desktop exception hierarchy
    (frontend/izvoria_desktop/exceptions/), never engine or third-party exceptions.
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
        evidence, or -- for Reasoned Answer -- a clarification question back at the user.
        """

    @abstractmethod
    def search(self, prompt: str, mode: SearchMode, limit: int) -> SearchOutcome:
        """Ranked retrieval, with or without the SLM relevance scrubber."""

    @abstractmethod
    def languages(self) -> List[LanguageOption]:
        """Every language the appliance serves. Reads configuration only, so it is callable before boot."""

    @abstractmethod
    def active_language(self) -> str:
        """The language new batches are indexed in and answers are written in. Configuration only."""

    @abstractmethod
    def switch_language(self, code: str) -> None:
        """
        Makes `code` the active language by rebuilding the engine (seconds, no model load). Refused while
        a job holds the engine. A failed rebuild comes back up in the previous language, then raises.
        """

    @abstractmethod
    def ingest(self, path: str, mode: IngestMode, language: str) -> IngestionSummary:
        """
        Switches to `language` first when it isn't the active one: a batch is indexed in its own language.

        Runs a batch pipeline to completion. Very long and not interruptible; blocks live
        inference for its whole duration.
        """

    @abstractmethod
    def propose_report_plan(self, topic: str, mode: ReportMode) -> ReportPlanDraft:
        """
        Scopes a research report and returns the plan for the user to approve or edit.

        Short (about a minute). The long run is a separate call on purpose: a misread topic is cheap to
        fix here and expensive to fix after half an hour of GPU time.
        """
        pass

    @abstractmethod
    def run_report(self, plan: ReportPlanDraft, mode: ReportMode) -> ReportDocument:
        """
        Runs the approved plan. Takes the GPU exclusively for the whole run.

        A cancelled run returns a PARTIAL report rather than raising -- the sections already written are
        cited and valid, and the user asked to stop, which is not an error.
        """
        pass

    @abstractmethod
    def request_report_cancel(self, report_id: uuid.UUID) -> bool:
        """
        Asks a running report to stop, returning whether one was actually running.

        Called DIRECTLY from the GUI thread, not through the worker: the worker is busy running the very
        report this is trying to stop, so a queued call would sit behind it for the full run. Safe because
        the engine only sets a threading.Event here -- no model, no database, no blocking.
        """
        pass

    @abstractmethod
    def hydrate_evidence(self, chunk_ids: List[uuid.UUID]) -> List[Evidence]:
        """
        Fetches verbatim excerpts and clip paths for chunk ids. Needed because a citation
        carries ids and timestamps but no text. Cheap: an indexed read, no model, no VRAM.
        """

    @abstractmethod
    def resume_report(self, report_id: uuid.UUID) -> ReportDocument:
        """
        Writes the sections a stopped, failed or interrupted report left pending, from its stored
        evidence bank. A GPU job like run_report, but it gathers nothing. Switches to the report's own
        language first, so the remaining sections are written in the language the rest is in.
        """

    # ---------- history ----------
    #
    # Database reads and deletes: no model, no VRAM, no engine lock. The ONLY gateway methods that
    # may be called from the bridge's reader thread, concurrently with the worker thread -- that is
    # what keeps history open while an ingestion or a report runs for hours.

    @abstractmethod
    def list_conversations(self) -> List[ConversationRef]:
        """Past conversations, most recently active first."""

    @abstractmethod
    def load_conversation(self, session_id: uuid.UUID) -> List[ChatMessage]:
        """A conversation's turns as user/assistant messages, each answer with the evidence it cited."""

    @abstractmethod
    def delete_conversation(self, session_id: uuid.UUID) -> None:
        """Forgets a conversation."""

    @abstractmethod
    def list_searches(self) -> List[SearchRecordRef]:
        """Past searches, newest first."""

    @abstractmethod
    def load_search(self, record_id: uuid.UUID) -> SearchOutcome:
        """A past search with the hits it returned then (saved_at set)."""

    @abstractmethod
    def delete_search(self, record_id: uuid.UUID) -> None:
        """Forgets a past search."""

    @abstractmethod
    def list_reports(self) -> List[ReportRef]:
        """Stored reports, newest first."""

    @abstractmethod
    def load_report(self, report_id: uuid.UUID) -> ReportDocument:
        """A stored report rendered as it stands, finished or not."""

    @abstractmethod
    def load_report_plan(self, report_id: uuid.UUID) -> ReportPlanDraft:
        """The outline of a report still awaiting approval, back in the approval step."""

    def set_progress_listener(self, listener: Optional[Callable[..., None]]) -> None:
        """
        Registers a per-file pipeline progress callback. Optional: a gateway that cannot report
        progress may ignore it, so the UI degrades to its log pane instead of breaking.
        """
        return None
