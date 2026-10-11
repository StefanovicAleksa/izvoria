# Canvas: desktop_engine_worker_v1
# Path: frontend/izvoria_desktop/service/engine_worker.py

"""
The object that actually calls the engine, living on its own QThread.

Every request arrives as a queued signal, so Qt's event loop on this thread runs them strictly
one at a time. That serialization is not a nicety: acquiring a model evicts every other resident
one, so two overlapping GPU calls would fight over a single VRAM slot rather than queue politely.
"""

import logging
import uuid
from typing import List

from PySide6.QtCore import QObject, Signal, Slot

from izvoria_desktop.domain.entities import ProgressTick
from izvoria_desktop.domain.enums import ChatMode, IngestMode, SearchMode
from izvoria_desktop.domain.interfaces import IEngineGateway
from izvoria_desktop.exceptions.base import DesktopError
from izvoria_desktop.utils.error_mapper import map_engine_error

logger = logging.getLogger(__name__)


class EngineWorker(QObject):
    """
    Slots here run on the worker thread. Each emits exactly one terminal signal -- a result or a
    failure -- carrying the request id, so a view can match a reply to the request it made and
    ignore anything stale.
    """

    booted = Signal()
    chatReady = Signal(int, object)      # request_id, ChatMessage
    searchReady = Signal(int, object)    # request_id, SearchOutcome
    ingestReady = Signal(int, object)    # request_id, IngestionSummary
    evidenceReady = Signal(int, object)  # request_id, List[Evidence]
    reportPlanReady = Signal(int, object)  # request_id, ReportPlanDraft
    reportReady = Signal(int, object)      # request_id, ReportDocument
    languageSwitched = Signal(int, str)    # request_id, the language now active
    # The active language after any call that can change it: boot, a switch, an ingest in another language,
    # resuming a report written in another one. Emitted before that call's terminal signal.
    activeLanguage = Signal(str)
    progressed = Signal(object)          # ProgressTick
    failed = Signal(int, str, object)    # request_id, user_message, DesktopError
    engineStopped = Signal(int)            # request_id: the engine is down, the thread keeps running
    finished = Signal()

    def __init__(self, gateway: IEngineGateway):
        super().__init__()
        self._gateway = gateway
        self._shutting_down = False

    # ---------- lifecycle ----------

    @Slot()
    def boot(self) -> None:
        try:
            self._gateway.set_progress_listener(self._emit_progress)
            self._gateway.boot()
            self._emit_language()
            self.booted.emit()
        except Exception as exc:
            self._fail(0, exc, "Starting the appliance")

    @Slot()
    def shutdown(self) -> None:
        """
        Runs on the worker thread so the engine is stopped by the same thread that used it, then
        lets the thread's event loop exit. Emitted last, after VRAM is released.
        """
        self._shutting_down = True
        try:
            self._gateway.set_progress_listener(None)
            self._gateway.shutdown()
        except Exception as exc:
            logger.error(f"Shutdown raised and was swallowed so the window can close: {exc}")
        finally:
            self.finished.emit()

    @Slot(int)
    def stop_engine(self, request_id: int) -> None:
        """
        Stops the engine and frees its VRAM but, unlike shutdown(), keeps this thread's event loop alive so
        the appliance can be started again from the same window.
        """
        try:
            self._gateway.shutdown()
            self.engineStopped.emit(request_id)
        except Exception as exc:
            self._fail(request_id, exc, "Stopping the appliance")

    # ---------- work ----------

    @Slot(int, str, object, object, int)
    def run_chat(
        self,
        request_id: int,
        prompt: str,
        session_id: uuid.UUID,
        mode: ChatMode,
        limit: int,
    ) -> None:
        try:
            message = self._gateway.chat(prompt, session_id, mode, limit)
            self.chatReady.emit(request_id, message)
        except Exception as exc:
            self._fail(request_id, exc, "Chat")

    @Slot(int, str, object, int)
    def run_search(self, request_id: int, prompt: str, mode: SearchMode, limit: int) -> None:
        try:
            outcome = self._gateway.search(prompt, mode, limit)
            self.searchReady.emit(request_id, outcome)
        except Exception as exc:
            self._fail(request_id, exc, "Search")

    @Slot(int, str, object, str)
    def run_ingest(self, request_id: int, path: str, mode: IngestMode, language: str) -> None:
        try:
            summary = self._gateway.ingest(path, mode, language)
            self._emit_language()
            self.ingestReady.emit(request_id, summary)
        except Exception as exc:
            self._emit_language()
            self._fail(request_id, exc, "Ingestion")

    @Slot(int, str)
    def run_switch_language(self, request_id: int, code: str) -> None:
        try:
            self._gateway.switch_language(code)
            self._emit_language()
            self.languageSwitched.emit(request_id, self._gateway.active_language())
        except Exception as exc:
            # A failed switch comes back up in the previous language; say which one that is
            self._emit_language()
            self._fail(request_id, exc, "Switching language")

    @Slot(int, str, object)
    def run_report_plan(self, request_id: int, topic: str, mode) -> None:
        try:
            plan = self._gateway.propose_report_plan(topic, mode)
            self.reportPlanReady.emit(request_id, plan)
        except Exception as exc:
            self._fail(request_id, exc, "Planning a research report")

    @Slot(int, object, object)
    def run_report(self, request_id: int, plan, mode) -> None:
        """
        The long one -- tens of minutes, holding the GPU. A cancelled run still comes back through
        reportReady with a partial report, because the user stopping is not a failure.
        """
        try:
            report = self._gateway.run_report(plan, mode)
            self.reportReady.emit(request_id, report)
        except Exception as exc:
            self._fail(request_id, exc, "Research report")

    @Slot(int, object)
    def run_resume_report(self, request_id: int, report_id: uuid.UUID) -> None:
        """Finishes a stopped report from its evidence bank; comes back through reportReady like a run."""
        try:
            report = self._gateway.resume_report(report_id)
            self._emit_language()
            self.reportReady.emit(request_id, report)
        except Exception as exc:
            self._emit_language()
            self._fail(request_id, exc, "Resuming a research report")

    @Slot(int, object)
    def run_hydrate(self, request_id: int, chunk_ids: List[uuid.UUID]) -> None:
        try:
            evidence = self._gateway.hydrate_evidence(list(chunk_ids))
            self.evidenceReady.emit(request_id, evidence)
        except Exception as exc:
            self._fail(request_id, exc, "Loading cited excerpts")

    # ---------- helpers ----------

    def _emit_language(self) -> None:
        try:
            self.activeLanguage.emit(self._gateway.active_language())
        except Exception as exc:
            logger.error(f"Could not read the active language: {exc}")

    def _emit_progress(self, tick: ProgressTick) -> None:
        """
        Called by the gateway from inside the engine call, on this same thread. Emitting a signal
        is the only safe thing to do here: the connection to the GUI thread is queued, so the
        payload is handed over through the event loop rather than touching a widget directly.
        """
        self.progressed.emit(tick)

    def _fail(self, request_id: int, exc: Exception, context: str) -> None:
        error: DesktopError = (
            exc if isinstance(exc, DesktopError) else map_engine_error(exc, context=context)
        )
        logger.error(f"{context} failed: {error.user_message}", exc_info=error.original_exception)
        self.failed.emit(request_id, error.user_message, error)
