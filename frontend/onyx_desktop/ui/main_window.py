# Canvas: desktop_main_window_v1
# Path: frontend/onyx_desktop/ui/main_window.py

"""
The shell: sidebar, the four views, the status bar, starting and stopping the appliance, and the
shutdown choreography.

The close path is the subtle part. Blocking the GUI thread inside `closeEvent` to wait for the
engine would stop the event loop that has to deliver the worker's reply, so the close is deferred:
ignore the event, show that we are stopping, and close again once the engine is actually down.
"""

import logging
import uuid
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QActionGroup, QCloseEvent
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
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from onyx_desktop.domain.entities import (
    ChatMessage,
    IngestionSummary,
    ProgressTick,
    SearchOutcome,
    StatusSnapshot,
)
from onyx_desktop.domain.enums import ApplianceStatus, ChatMode, IngestMode, SearchMode
from onyx_desktop.service.engine_bridge import EngineBridge
from onyx_desktop.service.log_relay import LogRelay
from onyx_desktop.ui import appearance, icons, theme
from onyx_desktop.ui import translate as tl
from onyx_desktop.ui.views.chat_view import ChatView
from onyx_desktop.ui.views.ingest_view import IngestView
from onyx_desktop.ui.views.report_view import ReportView
from onyx_desktop.ui.views.search_view import SearchView
from onyx_desktop.ui.widgets.status_bar import StatusBar, status_label
from onyx_desktop.utils import i18n
from onyx_desktop.utils.i18n import N_, tr

logger = logging.getLogger(__name__)

# The nav button id IS the QStackedWidget index, so this order must match the views added in _build.
_NAV = (N_("Chat"), N_("Search"), N_("Ingest"), N_("Report"))


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

        tl.title(self, lambda: "Onyx" + (f"  —  {tr('stub mode')}" if stub_mode else ""))
        self.resize(1440, 900)

        self._build()
        self._paint_icons()
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
        frame.setFixedWidth(228)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 22, 0, 16)
        layout.setSpacing(2)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(20, 0, 14, 0)
        brand_box = QVBoxLayout()
        brand_box.setSpacing(2)
        brand = QLabel("Onyx")
        brand.setProperty("role", "brand")
        brand_box.addWidget(brand)
        brand_box.addWidget(tl.label("On-premise appliance", role="subtle"))
        brand_row.addLayout(brand_box, stretch=1)

        # The display language: what the app's own text is written in. Separate from "Answer in" below, which
        # is the appliance's language; like the light/dark toggle, it is pure presentation and works any time.
        self._display_language = QToolButton()
        self._display_language.setObjectName("ModeToggle")
        self._display_language.setCursor(Qt.CursorShape.PointingHandCursor)
        self._display_language.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self._display_language)
        group = QActionGroup(menu)
        group.setExclusive(True)
        self._display_actions = {}
        for language in i18n.available():
            action = menu.addAction(language.name)
            action.setCheckable(True)
            action.setChecked(language.code == i18n.current())
            action.triggered.connect(lambda _checked=False, code=language.code: self.set_display_language(code))
            group.addAction(action)
            self._display_actions[language.code] = action
        self._display_language.setMenu(menu)
        tl.text(self._display_language, lambda: i18n.current().upper())
        tl.tooltip(self._display_language, "Display language")
        brand_row.addWidget(self._display_language, alignment=Qt.AlignmentFlag.AlignTop)

        # Pure presentation, so it works at any time -- mid-ingestion included
        self._mode_toggle = QToolButton()
        self._mode_toggle.setObjectName("ModeToggle")
        self._mode_toggle.setIconSize(icons.ICON_SIZE)
        self._mode_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mode_toggle.clicked.connect(self._on_mode_toggled)
        brand_row.addWidget(self._mode_toggle, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(brand_row)
        layout.addSpacing(26)

        section = tl.label(lambda: tr("Workspace").upper(), role="section")
        section.setContentsMargins(20, 0, 20, 6)
        layout.addWidget(section)

        self._nav = QButtonGroup(self)
        self._nav.setExclusive(True)
        for index, label in enumerate(_NAV):
            button = QPushButton()
            tl.text(button, label)
            button.setProperty("role", "nav")
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self._nav.addButton(button, index)
            layout.addWidget(button)

        layout.addStretch(1)
        layout.addWidget(self._appliance_card())

        if self._stub_mode:
            # Developer-only (--stub), so it is not translated
            note = tl.label(tl.plain("Stub gateway — no GPU, no database, canned answers."), role="subtle")
            note.setWordWrap(True)
            note.setContentsMargins(20, 10, 20, 0)
            layout.addWidget(note)

        return frame

    def _appliance_card(self) -> QWidget:
        """The appliance's own controls, grouped: what it is doing, its answer language, start and stop."""
        holder = QWidget()
        holder.setObjectName("SidebarBlock")
        outer = QVBoxLayout(holder)
        outer.setContentsMargins(12, 0, 12, 0)

        card = QFrame()
        card.setObjectName("ApplianceCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)

        state_row = QHBoxLayout()
        state_row.setSpacing(8)
        self._card_dot = QLabel("●")
        state_row.addWidget(self._card_dot)
        self._card_state = QLabel()
        self._card_state.setProperty("role", "strong")
        state_row.addWidget(self._card_state, stretch=1)
        layout.addLayout(state_row)
        layout.addSpacing(2)

        answer_label = tl.label("Answer in")
        answer_label.setProperty("role", "subtle")
        layout.addWidget(answer_label)
        self._language = QComboBox()
        for option in self._bridge.languages:
            self._language.addItem(option.name, option.code)
        tl.tooltip(self._language, "The language answers and reports are written in. Search covers material in "
                                   "every language, and excerpts are always quoted in their own. Switching takes a "
                                   "few seconds.")
        # Switching rebuilds the engine, so it waits until one is running
        self._language.setEnabled(False)
        self._language.currentIndexChanged.connect(self._on_language_chosen)
        layout.addWidget(self._language)
        layout.addSpacing(4)

        self._start_button = QPushButton()
        tl.text(self._start_button, "Start appliance")
        self._start_button.setProperty("role", "primary")
        self._start_button.clicked.connect(self._bridge.boot)
        layout.addWidget(self._start_button)

        self._stop_button = QPushButton()
        tl.text(self._stop_button, "Stop appliance")
        self._stop_button.setProperty("role", "danger-outline")
        self._stop_button.clicked.connect(self._on_stop_clicked)
        self._stop_button.setVisible(False)
        layout.addWidget(self._stop_button)

        outer.addWidget(card)
        self._apply_card_status(self._bridge.status)
        return holder

    def _paint_icons(self) -> None:
        """The mode toggle's icon is a pixmap in the palette's colours, so it is redrawn whenever the mode changes."""
        p = theme.PALETTE
        # Shows where a click goes, not where you are
        target = "light" if theme.MODE == "dark" else "dark"
        self._mode_toggle.setIcon(icons.nav_icon("sun" if target == "light" else "moon",
                                                 p.sidebar_muted, p.sidebar_text))
        tl.tooltip(self._mode_toggle, "Switch to light mode" if target == "light" else "Switch to dark mode")

    @Slot()
    def _on_mode_toggled(self) -> None:
        self.set_mode("light" if theme.MODE == "dark" else "dark")

    def set_mode(self, mode: str) -> None:
        """
        Switches light/dark in place. The stylesheet repaints most of the window; the few widgets that bake a
        colour into rich text or a pixmap at build time expose apply_theme() and redraw themselves.
        """
        if mode == theme.MODE:
            return
        theme.use(mode)
        theme.apply_to(QApplication.instance())
        self._paint_icons()
        self._apply_card_status(self._bridge.status)
        for widget in self.findChildren(QWidget):
            apply_theme = getattr(widget, "apply_theme", None)
            if callable(apply_theme):
                apply_theme()
        appearance.save_mode(mode)

    def set_display_language(self, code: str) -> None:
        """
        Switches the language of the app's own text in place, at any time. Every string was set through
        `ui/translate.py`, which replays it in the new language -- headings, dynamic lines and history alike.
        The answer language is untouched: it belongs to the appliance, and has its own picker.
        """
        if code == i18n.current():
            return
        i18n.use(code)
        tl.retranslate_all()
        for language_code, action in self._display_actions.items():
            action.setChecked(language_code == code)
        appearance.save_display_language(code)

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
        self._bridge.stopped.connect(self._on_stopped)
        self._bridge.chatFinished.connect(self._on_chat_finished)
        self._bridge.searchFinished.connect(self._on_search_finished)
        self._bridge.ingestFinished.connect(self._on_ingest_finished)
        self._bridge.reportPlanFinished.connect(self._on_report_plan_finished)
        self._bridge.reportFinished.connect(self._on_report_finished)
        self._bridge.progressTicked.connect(self._on_progress)
        self._bridge.statusChanged.connect(self._on_status)
        self._bridge.busyChanged.connect(self._on_busy)
        self._bridge.failed.connect(self._on_failed)


    # ---------- engine events ----------

    @Slot()
    def _on_booted(self) -> None:
        self._language.setEnabled(not self._bridge.is_busy)
        self._refresh_appliance_controls()
        self._refresh_history()

    @Slot()
    def _on_stop_clicked(self) -> None:
        self._bridge.stop_appliance()

    @Slot()
    def _on_stopped(self) -> None:
        """The engine is down but the window stays: what is on screen stays readable, history waits for a start."""
        tl.text(self._start_button, "Start appliance")
        self._language.setEnabled(False)
        for view in (self._chat, self._search, self._report):
            view.set_history_unavailable(N_("Start the appliance to see history."))
        self._refresh_appliance_controls()

    def _refresh_appliance_controls(self) -> None:
        """
        Start while the engine is down, Stop while it is up. Stop is refused, not merely hidden, while
        anything runs -- a chat turn, a search, an ingestion, a report, a language switch -- because each of
        them holds the engine it would tear down.
        """
        status = self._bridge.status
        engine_down = status in (ApplianceStatus.OFFLINE, ApplianceStatus.ERROR)
        self._start_button.setVisible(engine_down)
        self._start_button.setEnabled(engine_down)
        self._stop_button.setVisible(not engine_down)
        can_stop = self._bridge.can_stop
        self._stop_button.setEnabled(can_stop)
        tl.tooltip(
            self._stop_button,
            "Stops the appliance. Start it again at any time."
            if can_stop else
            "Wait for the current job to finish before stopping the appliance.",
        )
        self._apply_card_status(status)

    def _apply_card_status(self, status: ApplianceStatus) -> None:
        colour = theme.STATUS_COLOURS.get(status.value, theme.TEXT_MUTED)
        self._card_dot.setStyleSheet(f"color: {colour}; font-size: 13px;")
        tl.text(self._card_state, lambda: status_label(status))

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
            listing.set_history_unavailable(tr("History could not be loaded: {reason}", reason=message))
            return
        owner = self._chat if "conversation" in method else self._search if "search" in method else self._report
        owner.show_failure(message)

    @Slot(object)
    def _on_status(self, snapshot: StatusSnapshot) -> None:
        self._status.apply(snapshot)
        if snapshot.status is ApplianceStatus.ERROR:
            tl.text(self._start_button, "Retry start")
        self._refresh_appliance_controls()

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
        self._refresh_appliance_controls()
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
        box.setWindowTitle(tr("Report in progress") if researching else tr("Adding material"))
        box.setText(
            tr("A report is being written.")
            if researching else
            tr("Material is being added and this cannot be interrupted.")
        )
        box.setInformativeText(tr(
            "Quitting now stops it partway. Some files may be left half-added and will need to be "
            "added again."
        ))
        keep = box.addButton(tr("Keep running"), QMessageBox.ButtonRole.RejectRole)
        box.addButton(tr("Quit anyway"), QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(keep)
        box.exec()
        return box.clickedButton() is not keep
