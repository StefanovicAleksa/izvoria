# Canvas: desktop_main_window_v1
# Path: frontend/onyx_desktop/ui/main_window.py

"""
The shell: sidebar, the three views, the status bar, and the shutdown choreography.

The close path is the subtle part. Blocking the GUI thread inside `closeEvent` to wait for the
engine would stop the event loop that has to deliver the worker's reply, so the close is deferred:
ignore the event, show that we are stopping, and close again once the engine is actually down.
"""

import logging
import uuid
from typing import Optional

from PySide6.QtCore import QTimer, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from onyx_desktop.domain.entities import (
    ChatMessage,
    IngestionSummary,
    LogLine,
    ProgressTick,
    SearchOutcome,
    StatusSnapshot,
)
from onyx_desktop.domain.enums import ApplianceStatus, ChatMode, IngestMode, SearchMode
from onyx_desktop.service.engine_bridge import EngineBridge
from onyx_desktop.service.log_relay import LogRelay
from onyx_desktop.ui.views.chat_view import ChatView
from onyx_desktop.ui.views.ingest_view import IngestView
from onyx_desktop.ui.views.report_view import ReportView
from onyx_desktop.ui.views.search_view import SearchView
from onyx_desktop.ui.widgets.status_bar import StatusBar

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self,
        bridge: EngineBridge,
        log_relay: LogRelay,
        stub_mode: bool = False,
    ):
        super().__init__()
        self._bridge = bridge
        self._relay = log_relay
        self._stub_mode = stub_mode
        self._shutdown_done = False
        self._closing = False

        self.setWindowTitle("Onyx" + ("  —  stub mode" if stub_mode else ""))
        self.resize(1440, 900)

        self._build()
        self._connect()
        self._ingest.set_languages(self._bridge.languages, self._bridge.active_language)
        self._show_active_language(self._bridge.active_language)

    # ---------- construction ----------

    def _build(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        body.addWidget(self._sidebar())

        self._views = QStackedWidget()
        self._chat = ChatView()
        self._search = SearchView()
        self._ingest = IngestView()
        self._report = ReportView()
        for view in (self._chat, self._search, self._ingest, self._report):
            self._views.addWidget(view)
        body.addWidget(self._views, stretch=1)

        root.addLayout(body, stretch=1)

        self._status = StatusBar()
        root.addWidget(self._status)

        self.setCentralWidget(central)

    def _sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Sidebar")
        frame.setFixedWidth(210)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(6)

        brand = QLabel("ONYX")
        brand.setStyleSheet("font-size: 17px; font-weight: 700; letter-spacing: 3px;")
        layout.addWidget(brand)

        subtitle = QLabel("On-premise appliance")
        subtitle.setProperty("role", "subtle")
        layout.addWidget(subtitle)
        layout.addSpacing(18)

        self._nav = QButtonGroup(self)
        self._nav.setExclusive(True)
        for index, label in enumerate(("Chat", "Search", "Ingest", "Report")):
            button = QPushButton(label)
            button.setProperty("role", "nav")
            button.setCheckable(True)
            button.setChecked(index == 0)
            self._nav.addButton(button, index)
            layout.addWidget(button)

        layout.addStretch(1)

        answer_label = QLabel("Answer in")
        answer_label.setProperty("role", "subtle")
        layout.addWidget(answer_label)
        self._language = QComboBox()
        for option in self._bridge.languages:
            self._language.addItem(option.name, option.code)
        self._language.setToolTip(
            "The language answers, reports and verified searches are written in. Search finds evidence in "
            "every language, and excerpts are always quoted in their own. Switching restarts the appliance "
            "in seconds; it loads no model."
        )
        # Switching rebuilds the engine, so it waits until one is running
        self._language.setEnabled(False)
        self._language.currentIndexChanged.connect(self._on_language_chosen)
        layout.addWidget(self._language)
        layout.addSpacing(8)

        self._start_button = QPushButton("Start appliance")
        self._start_button.setProperty("role", "primary")
        self._start_button.clicked.connect(self._bridge.boot)
        layout.addWidget(self._start_button)

        if self._stub_mode:
            note = QLabel("Stub gateway — no GPU, no database, canned answers.")
            note.setProperty("role", "subtle")
            note.setWordWrap(True)
            layout.addWidget(note)

        return frame

    def _connect(self) -> None:
        self._nav.idClicked.connect(self._views.setCurrentIndex)

        self._chat.chatRequested.connect(self._on_chat_requested)
        self._search.searchRequested.connect(self._on_search_requested)
        self._ingest.ingestRequested.connect(self._on_ingest_requested)
        self._report.planRequested.connect(self._on_report_plan_requested)
        self._report.reportRequested.connect(self._on_report_requested)
        self._report.cancelRequested.connect(self._on_report_cancel_requested)
        self._report.resumeRequested.connect(self._on_report_resume_requested)

        # History: views ask, the bridge answers from its reader thread.
        self._chat.conversationChosen.connect(self._bridge.load_conversation)
        self._chat.conversationDeleteRequested.connect(self._on_conversation_delete_requested)
        self._search.savedSearchChosen.connect(self._bridge.load_search)
        self._search.savedSearchDeleteRequested.connect(self._bridge.delete_search)
        self._report.savedReportRequested.connect(self._bridge.load_report)
        self._report.savedPlanRequested.connect(self._bridge.load_report_plan)
        self._bridge.conversationsListed.connect(self._chat.set_conversations)
        self._bridge.conversationLoaded.connect(self._chat.show_conversation)
        self._bridge.searchesListed.connect(self._search.set_saved_searches)
        self._bridge.searchLoaded.connect(self._search.show_results)
        self._bridge.reportsListed.connect(self._report.set_reports)
        self._bridge.reportLoaded.connect(self._report.show_report)
        self._bridge.reportPlanLoaded.connect(self._report.show_plan)
        self._bridge.historyFailed.connect(self._on_history_failed)

        self._bridge.languageChanged.connect(self._show_active_language)
        self._bridge.booted.connect(self._on_booted)
        self._bridge.chatFinished.connect(self._on_chat_finished)
        self._bridge.searchFinished.connect(self._on_search_finished)
        self._bridge.ingestFinished.connect(self._on_ingest_finished)
        self._bridge.reportPlanFinished.connect(self._on_report_plan_finished)
        self._bridge.reportFinished.connect(self._on_report_finished)
        self._bridge.progressTicked.connect(self._on_progress)
        self._bridge.statusChanged.connect(self._on_status)
        self._bridge.busyChanged.connect(self._on_busy)
        self._bridge.failed.connect(self._on_failed)

        self._relay.lineLogged.connect(self._on_log_line)
        self._relay.phaseChanged.connect(self._ingest.set_phase_from_log)

    # ---------- engine events ----------

    @Slot()
    def _on_booted(self) -> None:
        self._start_button.setVisible(False)
        self._language.setEnabled(not self._bridge.is_busy)
        self._refresh_history()

    @Slot(str)
    def _show_active_language(self, code: str) -> None:
        """Every picker and header follows the engine's language, including a switch an ingest or resume made."""
        index = self._language.findData(code)
        if index >= 0 and index != self._language.currentIndex():
            self._language.blockSignals(True)
            self._language.setCurrentIndex(index)
            self._language.blockSignals(False)
        name = self._bridge.language_name(code)
        for view in (self._chat, self._search, self._report):
            view.set_answer_language(name)
        self._ingest.select_language(code)

    @Slot(int)
    def _on_language_chosen(self, _index: int) -> None:
        # Rebuilt from the combo's QVariant at the boundary, like every other combo in the app
        code = str(self._language.currentData())
        if self._bridge.switch_language(code) is None:
            # Refused (busy) or already active: the picker goes back to what the engine actually runs
            self._show_active_language(self._bridge.active_language)

    def _refresh_history(self) -> None:
        """Each list is one indexed read; refreshing all three after any change keeps them simple and right."""
        if not self._bridge.history_available:
            return
        self._bridge.list_conversations()
        self._bridge.list_searches()
        self._bridge.list_reports()

    @Slot(str, str)
    def _on_history_failed(self, method: str, message: str) -> None:
        listing = {
            "list_conversations": self._chat,
            "list_searches": self._search,
            "list_reports": self._report,
        }.get(method)
        if listing is not None:
            listing.set_history_unavailable(f"History could not be loaded: {message}")
            return
        owner = self._chat if "conversation" in method else self._search if "search" in method else self._report
        owner.show_failure(message)

    @Slot(object)
    def _on_status(self, snapshot: StatusSnapshot) -> None:
        self._status.apply(snapshot)
        if snapshot.status is ApplianceStatus.ERROR:
            self._start_button.setVisible(True)
            self._start_button.setText("Retry start")

    @Slot(bool)
    def _on_busy(self, busy: bool) -> None:
        # Ingestion locks the hardware, so live views must refuse input for its whole duration,
        # not merely look disabled.
        ingesting = self._bridge.is_ingesting
        self._chat.set_busy(busy)
        self._search.set_busy(busy)
        self._ingest.set_busy(busy)
        self._report.set_busy(busy)
        self._language.setEnabled(not busy and self._bridge.status is not ApplianceStatus.OFFLINE
                                  and self._bridge.status is not ApplianceStatus.ERROR)
        if ingesting:
            self._views.setCurrentWidget(self._ingest)
        elif self._bridge.is_researching:
            # A report holds the GPU for tens of minutes. Showing its progress beats leaving the user
            # on a chat view that silently refuses to answer.
            self._views.setCurrentWidget(self._report)

    @Slot(int, object)
    def _on_chat_finished(self, _request_id: int, message: ChatMessage) -> None:
        self._chat.add_message(message)
        self._bridge.list_conversations()

    @Slot(int, object)
    def _on_search_finished(self, _request_id: int, outcome: SearchOutcome) -> None:
        self._search.show_results(outcome)
        self._bridge.list_searches()

    @Slot(int, object)
    def _on_ingest_finished(self, _request_id: int, summary: IngestionSummary) -> None:
        self._ingest.show_summary(summary)

    @Slot(object)
    def _on_progress(self, tick: ProgressTick) -> None:
        # Routed, not broadcast. One listener sees every job in the process, so a report's ticks would
        # otherwise drive the ingestion counters and vice versa.
        if self._bridge.is_researching:
            self._report.apply_tick(tick)
        else:
            self._ingest.apply_tick(tick)

    @Slot(object)
    def _on_log_line(self, line: LogLine) -> None:
        self._ingest.append_log(line)

    @Slot(int, str)
    def _on_failed(self, _request_id: int, message: str) -> None:
        current = self._views.currentWidget()
        if current is self._chat:
            self._chat.show_failure(message)
        elif current is self._search:
            self._search.show_failure(message)
        elif current is self._ingest:
            self._ingest.show_failure(message)
        elif current is self._report:
            self._report.show_failure(message)
        self._status.apply(StatusSnapshot(status=self._bridge.status, detail=message))
        # A failed run still changed something worth listing (a report now FAILED, for one).
        self._refresh_history()

    @Slot(int, object)
    def _on_report_plan_finished(self, _request_id: int, plan) -> None:
        self._report.show_plan(plan)
        self._bridge.list_reports()

    @Slot(int, object)
    def _on_report_finished(self, _request_id: int, report) -> None:
        self._report.show_report(report)
        self._bridge.list_reports()

    # ---------- view requests ----------

    @Slot(str, object, object, int)
    def _on_chat_requested(
        self, prompt: str, session_id: uuid.UUID, mode: ChatMode, limit: int
    ) -> None:
        self._bridge.send_chat(prompt, session_id, mode, limit)

    @Slot(object)
    def _on_conversation_delete_requested(self, session_id: uuid.UUID) -> None:
        if self._bridge.delete_conversation(session_id) is not None:
            self._chat.forget_conversation(session_id)

    @Slot(str, object, int)
    def _on_search_requested(self, prompt: str, mode: SearchMode, limit: int) -> None:
        self._bridge.run_search(prompt, mode, limit)

    @Slot(str, object)
    def _on_report_plan_requested(self, topic: str, mode) -> None:
        # Mirrors the ingest path: the view only switches state once the request is accepted, so a
        # refusal (appliance busy) leaves the outline the user was working with intact.
        if self._bridge.plan_report(topic, mode) is not None:
            self._report.start_planning(topic)

    @Slot(object, object)
    def _on_report_requested(self, plan, mode) -> None:
        if self._bridge.run_report(plan, mode) is not None:
            self._report.start_run(plan)

    @Slot(object)
    def _on_report_cancel_requested(self, report_id) -> None:
        """
        Straight to the bridge, which calls the gateway on THIS thread rather than queueing.

        The worker is busy running the report being stopped, so a queued cancel would only arrive once
        the run it was meant to interrupt had already finished.
        """
        self._bridge.request_report_cancel(report_id)

    @Slot(object)
    def _on_report_resume_requested(self, report_id) -> None:
        if self._bridge.resume_report(report_id) is not None:
            self._report.start_resume(report_id)

    @Slot(str, object, str)
    def _on_ingest_requested(self, path: str, mode: IngestMode, language: str) -> None:
        if self._bridge.start_ingest(path, mode, language) is not None:
            self._ingest.start_run()

    # ---------- shutdown ----------

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._shutdown_done:
            event.accept()
            return

        if (self._bridge.is_ingesting or self._bridge.is_researching) and not self._confirm_force_quit():
            event.ignore()
            return

        if self._closing:
            # A second close while the first is still in progress: let it through rather than
            # trapping the user in a window that will not shut.
            event.accept()
            return

        # Refuse this close and finish the job on the next turn of the event loop.
        #
        # Two reasons it cannot be done inline. Calling self.close() from inside closeEvent is
        # reentrant and leaves the widget half-closed -- the event loop then never exits. And
        # stopping the engine blocks the GUI thread, so doing it here freezes the window mid-close
        # instead of showing that it is shutting down.
        self._closing = True
        event.ignore()
        self._status.apply(StatusSnapshot(
            status=ApplianceStatus.OFFLINE, detail="Stopping the appliance…"
        ))
        self.setEnabled(False)
        self._relay.detach()
        QTimer.singleShot(0, self._finish_shutdown)

    def _finish_shutdown(self) -> None:
        """Runs after closeEvent has unwound, so close() here is a fresh, non-reentrant close."""
        finished = self._bridge.stop(timeout_ms=30000)
        if not finished:
            logger.warning("Engine did not stop cleanly; closing anyway.")
        self._shutdown_done = True
        self.close()
        # Don't rely on quitOnLastWindowClosed: a stray dialog or a hidden top-level window would
        # keep the loop alive and the process would never exit.
        QApplication.instance().quit()

    def _confirm_force_quit(self) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        researching = self._bridge.is_researching
        box.setWindowTitle("Research in progress" if researching else "Ingestion in progress")
        box.setText(
            "A research report is running and holds the GPU."
            if researching else
            "An ingestion is running and cannot be interrupted."
        )
        box.setInformativeText(
            "The pipeline has no stop point — it is inside native transcription and LLM "
            "calls that ignore interruption. Quitting now ends the process mid-file and may "
            "leave partially indexed material in the vault, with no way to resume."
        )
        keep = box.addButton("Keep running", QMessageBox.ButtonRole.RejectRole)
        box.addButton("Quit anyway", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(keep)
        box.exec()
        return box.clickedButton() is not keep
