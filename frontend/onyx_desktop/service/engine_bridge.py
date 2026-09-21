# Canvas: desktop_engine_bridge_v1
# Path: frontend/onyx_desktop/service/engine_bridge.py

"""
The GUI thread's single point of contact with the appliance.

Views never see the worker, the gateway or anything from `app.*`. They call request methods here
and connect to the signals here. Requests cross to the worker thread as queued signal emissions,
which is what makes them serialize: one engine call at a time, matching a VRAM pool that holds
one model at a time.
"""

import itertools
import logging
import uuid
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal

from onyx_desktop.domain.entities import StatusSnapshot
from onyx_desktop.domain.enums import ApplianceStatus, ChatMode, IngestMode, SearchMode
from onyx_desktop.domain.interfaces import IEngineGateway
from onyx_desktop.service.engine_worker import EngineWorker

logger = logging.getLogger(__name__)


class EngineBridge(QObject):
    """
    Owns the worker thread and tracks what is in flight.

    `busyChanged(True)` is the UI's interlock: while a call is running the composer and the search
    button are disabled, so a double-click cannot enqueue a second GPU call behind the first.
    """

    # Requests, connected to worker slots with a queued connection.
    _bootRequested = Signal()
    _chatRequested = Signal(int, str, object, object, int)
    _searchRequested = Signal(int, str, object, int)
    _ingestRequested = Signal(int, str, object)
    _hydrateRequested = Signal(int, object)
    _shutdownRequested = Signal()

    # Results, for views to connect to.
    booted = Signal()
    chatFinished = Signal(int, object)
    searchFinished = Signal(int, object)
    ingestFinished = Signal(int, object)
    evidenceFinished = Signal(int, object)
    progressTicked = Signal(object)
    failed = Signal(int, str)
    busyChanged = Signal(bool)
    statusChanged = Signal(object)

    def __init__(self, gateway: IEngineGateway, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._ids = itertools.count(1)
        self._pending: Dict[int, str] = {}
        self._status = ApplianceStatus.OFFLINE
        self._stopped = False

        self._thread = QThread()
        self._thread.setObjectName("onyx-engine")
        self._worker = EngineWorker(gateway)
        self._worker.moveToThread(self._thread)

        # Requests in. Queued because emitter and receiver live on different threads.
        self._bootRequested.connect(self._worker.boot)
        self._chatRequested.connect(self._worker.run_chat)
        self._searchRequested.connect(self._worker.run_search)
        self._ingestRequested.connect(self._worker.run_ingest)
        self._hydrateRequested.connect(self._worker.run_hydrate)
        self._shutdownRequested.connect(self._worker.shutdown)

        # Results out.
        self._worker.booted.connect(self._on_booted)
        self._worker.chatReady.connect(lambda rid, msg: self._settle(rid, self.chatFinished, msg))
        self._worker.searchReady.connect(lambda rid, res: self._settle(rid, self.searchFinished, res))
        self._worker.ingestReady.connect(self._on_ingest_done)
        self._worker.evidenceReady.connect(lambda rid, ev: self._settle(rid, self.evidenceFinished, ev))
        self._worker.progressed.connect(self.progressTicked)
        self._worker.failed.connect(self._on_failed)

        # DirectConnection, deliberately. `finished` is emitted on the worker thread while stop()
        # is blocked in thread.wait() on the GUI thread. A QThread object's affinity is the GUI
        # thread, so an auto connection would queue quit() behind a GUI event loop that is not
        # running -- a guaranteed deadlock until the timeout. QThread.quit() is documented
        # thread-safe, so invoking it directly on the worker thread is both correct and the only
        # thing that terminates the loop here.
        self._worker.finished.connect(
            self._thread.quit, Qt.ConnectionType.DirectConnection
        )
        self._thread.finished.connect(self._worker.deleteLater)

        self._thread.start()

    # ---------- state ----------

    @property
    def status(self) -> ApplianceStatus:
        return self._status

    @property
    def is_busy(self) -> bool:
        return bool(self._pending)

    @property
    def is_ingesting(self) -> bool:
        return "ingest" in self._pending.values()

    def _set_status(self, status: ApplianceStatus, detail: str = "") -> None:
        self._status = status
        self.statusChanged.emit(StatusSnapshot(status=status, detail=detail))

    # ---------- requests ----------

    def boot(self) -> None:
        if self._status is not ApplianceStatus.OFFLINE:
            return
        self._set_status(ApplianceStatus.BOOTING, "Loading the appliance…")
        self._bootRequested.emit()

    def send_chat(
        self,
        prompt: str,
        session_id: uuid.UUID,
        mode: ChatMode,
        limit: int = 5,
    ) -> Optional[int]:
        request_id = self._begin("chat", ApplianceStatus.WORKING, "Answering…")
        if request_id is None:
            return None
        self._chatRequested.emit(request_id, prompt, session_id, mode, limit)
        return request_id

    def run_search(self, prompt: str, mode: SearchMode, limit: int = 5) -> Optional[int]:
        request_id = self._begin("search", ApplianceStatus.WORKING, "Searching…")
        if request_id is None:
            return None
        self._searchRequested.emit(request_id, prompt, mode, limit)
        return request_id

    def start_ingest(self, path: str, mode: IngestMode) -> Optional[int]:
        request_id = self._begin(
            "ingest", ApplianceStatus.INGESTING, f"Ingesting {path}…"
        )
        if request_id is None:
            return None
        self._ingestRequested.emit(request_id, path, mode)
        return request_id

    def hydrate_evidence(self, chunk_ids: List[uuid.UUID]) -> Optional[int]:
        request_id = self._begin("hydrate", ApplianceStatus.WORKING, "Loading excerpts…")
        if request_id is None:
            return None
        self._hydrateRequested.emit(request_id, list(chunk_ids))
        return request_id

    def _begin(self, kind: str, status: ApplianceStatus, detail: str) -> Optional[int]:
        """
        Refuses a request the appliance cannot take right now, rather than letting it queue up
        behind a long job and surface as a confusing error minutes later.
        """
        if self._status is ApplianceStatus.OFFLINE or self._status is ApplianceStatus.BOOTING:
            self.failed.emit(0, "The appliance is not running yet.")
            return None
        if self._pending:
            self.failed.emit(0, "The appliance is already working on something.")
            return None

        request_id = next(self._ids)
        self._pending[request_id] = kind
        self.busyChanged.emit(True)
        self._set_status(status, detail)
        return request_id

    # ---------- results ----------

    def _on_booted(self) -> None:
        self._set_status(ApplianceStatus.READY, "Ready")
        self.booted.emit()

    def _on_ingest_done(self, request_id: int, summary) -> None:
        self._settle(request_id, self.ingestFinished, summary)

    def _settle(self, request_id: int, signal: Signal, payload) -> None:
        self._pending.pop(request_id, None)
        if not self._pending:
            self.busyChanged.emit(False)
            self._set_status(ApplianceStatus.READY, "Ready")
        signal.emit(request_id, payload)

    def _on_failed(self, request_id: int, message: str, error) -> None:
        was_boot = self._status is ApplianceStatus.BOOTING
        self._pending.pop(request_id, None)
        if not self._pending:
            self.busyChanged.emit(False)
            self._set_status(
                ApplianceStatus.ERROR if was_boot else ApplianceStatus.READY,
                message if was_boot else "Ready",
            )
        self.failed.emit(request_id, message)

    # ---------- teardown ----------

    def stop(self, timeout_ms: int = 30000) -> bool:
        """
        Stops the engine and joins the worker thread. Returns False if the thread did not finish
        in time, which in practice means an ingestion is still running: those cannot be
        interrupted, so the caller has to decide whether to keep waiting.
        """
        if self._stopped:
            return True
        self._stopped = True
        self._shutdownRequested.emit()
        finished = self._thread.wait(timeout_ms)
        if not finished:
            logger.warning(
                "Engine worker did not finish within %sms; a batch job is likely still running.",
                timeout_ms,
            )
        self._set_status(ApplianceStatus.OFFLINE, "Stopped")
        return finished
