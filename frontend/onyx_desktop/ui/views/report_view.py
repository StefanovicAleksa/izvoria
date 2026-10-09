# Canvas: desktop_report_view_v1
# Path: frontend/onyx_desktop/ui/views/report_view.py

"""
Deep Research Report: a topic in, a multi-page cited report out.

Three states in a stack -- compose, run, read -- because they are genuinely different screens rather
than one screen with things hidden. The right-hand pane is the same `SourcePanel` the chat and search
views use, so a citation opens exactly the way a chat citation does.

Unlike every other view here, a run takes tens of minutes and holds the GPU, so this one has a Stop
button and a determinate progress bar. The bar can be determinate because the plan says up front how
many steps it will take -- ingestion never knows that.
"""

import logging
import uuid
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from onyx_desktop.data.report_writer import default_save_path, save_markdown
from onyx_desktop.domain.entities import Evidence, ProgressTick, ReportDocument, ReportPlanDraft, ReportRef
from onyx_desktop.domain.enums import ReportMode, ReportRunStatus
from onyx_desktop.ui import theme
from onyx_desktop.ui import translate as tl
from onyx_desktop.ui.widgets.answer_language import AnswerLanguageLabel
from onyx_desktop.ui.widgets.history_list import HistoryEntry, HistoryList
from onyx_desktop.ui.widgets.outline_editor import OutlineEditor
from onyx_desktop.ui.widgets.report_document import ReportDocumentView
from onyx_desktop.ui.widgets.source_panel import SourcePanel
from onyx_desktop.utils.progress_stages import report_step
from onyx_desktop.utils.formatting import when
from onyx_desktop.utils.i18n import N_, tr, tr_n

logger = logging.getLogger(__name__)

_COMPOSE, _RUNNING, _REPORT = 0, 1, 2

# Job phases a report emits. Used to ignore ticks from anything else.
_REPORT_PHASES = {
    "research_scoping", "research_outline", "research_section", "research_summary",
    "rag_retrieval", "rag_reranking", "retrieval_scrubbing",
}

_STATUS_LABELS = {
    ReportRunStatus.COMPLETED: N_("Finished"),
    ReportRunStatus.AWAITING_APPROVAL: N_("Outline not approved"),
    ReportRunStatus.CANCELLED: N_("Stopped"),
    ReportRunStatus.FAILED: N_("Failed"),
    # Only one report runs at a time and never across app runs, so a listed row in either of these was
    # interrupted (the app closed or crashed mid-run).
    ReportRunStatus.GATHERING: N_("Interrupted"),
    ReportRunStatus.WRITING: N_("Interrupted"),
}


class ReportView(QWidget):
    planRequested = Signal(str, object)      # topic, ReportMode
    reportRequested = Signal(object, object)  # ReportPlanDraft, ReportMode
    cancelRequested = Signal(object)          # report_id
    savedReportRequested = Signal(object)     # report_id -> load it as it stands
    savedPlanRequested = Signal(object)       # report_id -> back to the approval step
    resumeRequested = Signal(object)          # report_id -> write the sections it left pending

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._plan: Optional[ReportPlanDraft] = None
        self._document: Optional[ReportDocument] = None
        self._running_report_id: Optional[uuid.UUID] = None
        self._refs: dict = {}
        self._steps_done = 0
        self._build()

    # ---------- construction ----------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._history = HistoryList(
            N_("Reports"), N_("No reports yet. Finished, stopped and unapproved reports are kept here."),
        )
        self._reports: List[ReportRef] = []
        self._history.itemChosen.connect(self._on_history_chosen)
        splitter.addWidget(self._history)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(24, 20, 16, 20)
        left_layout.setSpacing(10)

        header = QHBoxLayout()
        header.addWidget(tl.label("Research report", role="heading"))
        self._answer_language = AnswerLanguageLabel()
        header.addWidget(self._answer_language)
        header.addStretch(1)
        left_layout.addLayout(header)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._compose_pane())
        self._stack.addWidget(self._running_pane())
        self._stack.addWidget(self._report_pane())
        left_layout.addWidget(self._stack, stretch=1)

        self._failure = QLabel()
        self._failure.setWordWrap(True)
        self._failure.setVisible(False)
        left_layout.addWidget(self._failure)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 20, 24, 20)
        right_layout.setSpacing(10)
        right_layout.addWidget(tl.label("Source review", role="heading"))
        hint = tl.label("Click a [Doc N] citation in the report to check it against its source.", role="subtle")
        hint.setWordWrap(True)
        right_layout.addWidget(hint)
        self._source_panel = SourcePanel()
        right_layout.addWidget(self._source_panel, stretch=1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([230, 620, 420])
        layout.addWidget(splitter)

    def _compose_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        blurb = tl.label(
            "Give a topic. The appliance proposes an outline for you to approve, then writes a cited report "
            "from your material.",
            role="subtle",
        )
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self._topic = QLineEdit()
        tl.placeholder(self._topic, "e.g. chain of custody for the evidence bag")
        self._topic.returnPressed.connect(self._on_plan)
        layout.addWidget(self._topic)

        controls = QHBoxLayout()
        self._mode = QComboBox()
        self._mode.addItem("", ReportMode.DEEP)
        self._mode.addItem("", ReportMode.EXHAUSTIVE)
        tl.items(self._mode, ["Standard", "Exhaustive"])
        tl.tooltip(self._mode, "Exhaustive looks harder for anything a section is missing. Slower, but when it "
                               "says a topic is not covered, the material really does not cover it.")
        controls.addWidget(self._mode, stretch=1)

        self._plan_button = QPushButton()
        tl.text(self._plan_button, "Plan research")
        self._plan_button.setProperty("role", "primary")
        self._plan_button.clicked.connect(self._on_plan)
        controls.addWidget(self._plan_button)
        layout.addLayout(controls)

        self._scoping = QLabel()
        self._scoping.setWordWrap(True)
        self._scoping.setProperty("role", "subtle")
        self._scoping.setVisible(False)
        layout.addWidget(self._scoping)

        self._outline = OutlineEditor()
        self._outline.accepted.connect(self._on_start)
        self._outline.discarded.connect(self._on_discard)
        self._outline.setVisible(False)
        layout.addWidget(self._outline, stretch=1)

        layout.addStretch(0)
        return pane

    def _running_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._running_topic = QLabel()
        self._running_topic.setWordWrap(True)
        layout.addWidget(self._running_topic)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        self._phase = tl.label("Starting…")
        self._phase.setWordWrap(True)
        self._phase.setProperty("role", "subtle")
        layout.addWidget(self._phase)

        note = tl.label("Chat and search are unavailable while the report is being written.")
        note.setWordWrap(True)
        note.setProperty("role", "subtle")
        layout.addWidget(note)

        self._stop = QPushButton()
        tl.text(self._stop, "Stop")
        tl.tooltip(self._stop, "Stops after the section being written now. Everything already written is kept.")
        self._stop.clicked.connect(self._on_stop)
        layout.addWidget(self._stop)

        layout.addStretch(1)
        return pane

    def _report_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._partial = QLabel()
        self._partial.setWordWrap(True)
        self._partial.setVisible(False)
        layout.addWidget(self._partial)

        self._body = ReportDocumentView()
        self._body.citationClicked.connect(self._on_citation)
        self._body.citationMissing.connect(self._on_citation_missing)
        layout.addWidget(self._body, stretch=1)

        buttons = QHBoxLayout()
        new_report = QPushButton()
        tl.text(new_report, "New report")
        new_report.clicked.connect(self._on_discard)
        buttons.addWidget(new_report)
        buttons.addStretch(1)

        self._resume = QPushButton()
        tl.text(self._resume, "Resume")
        tl.tooltip(self._resume, "Finishes the sections this report did not reach. The citations already written "
                                 "keep their numbers.")
        self._resume.setVisible(False)
        self._resume.clicked.connect(self._on_resume)
        buttons.addWidget(self._resume)

        self._save = QPushButton()
        tl.text(self._save, "Save as Markdown…")
        self._save.setProperty("role", "primary")
        self._save.clicked.connect(self._on_save)
        buttons.addWidget(self._save)
        layout.addLayout(buttons)
        return pane

    # ---------- public API ----------

    def set_answer_language(self, name: str) -> None:
        """A new report is written in the active language; a saved one keeps its own."""
        self._answer_language.set_language(name)

    def set_busy(self, busy: bool) -> None:
        self._plan_button.setEnabled(not busy)
        self._topic.setEnabled(not busy)
        self._mode.setEnabled(not busy)
        self._outline.set_busy(busy)
        self._resume.setEnabled(not busy)

    def set_reports(self, reports: List[ReportRef]) -> None:
        self._refs = {ref.report_id: ref for ref in reports}
        self._reports = list(reports)
        tl.on_retranslate(self, "history", self._render_history)
        if self._document is not None:
            self._sync_resume()

    def _render_history(self) -> None:
        self._history.set_entries([
            HistoryEntry(
                key=ref.report_id,
                title=ref.display_title,
                badge=tr(_STATUS_LABELS.get(ref.status, ref.status.value)),
                subtitle=when(ref.completed_at or ref.created_at),
            )
            for ref in self._reports
        ])
        if self._document is not None:
            self._history.select(self._document.report_id)

    def set_history_unavailable(self, text: str) -> None:
        self._history.set_unavailable(text)

    def start_resume(self, report_id: uuid.UUID) -> None:
        """Switches to the running pane for a resumed run, which writes sections but gathers nothing."""
        ref = self._refs.get(report_id)
        remaining = max(1, (ref.section_count if ref else 1) - (self._document.section_count if self._document else 0))
        self._running_report_id = report_id
        self._steps_done = 0
        self._progress.setRange(0, remaining + 1)  # one job per section left, plus the summary
        self._progress.setValue(0)
        tl.text(self._running_topic, "Resuming: {topic}", topic=ref.display_title if ref else "")
        tl.text(self._phase, "Finishing the remaining sections…")
        self._stop.setEnabled(True)
        tl.text(self._stop, "Stop")
        self._stack.setCurrentIndex(_RUNNING)

    def start_planning(self, topic: str) -> None:
        """
        Clears the previous outline while the new one is being scoped.

        Without this the old plan sits on screen for the whole scoping call and the new one swaps in
        underneath it. For the same topic the two look nearly identical, so nothing appears to happen
        and the approval step reads as though it were skipped.

        Called only once the bridge has ACCEPTED the request -- a refused one must not throw away an
        outline the user is still working with.
        """
        self._plan = None
        self._failure.setVisible(False)
        self._outline.setVisible(False)
        tl.text(self._scoping, "Drafting an outline for “{topic}”… This takes about a minute.", topic=topic)
        self._scoping.setVisible(True)
        self._stack.setCurrentIndex(_COMPOSE)

    def show_plan(self, plan: ReportPlanDraft) -> None:
        self._plan = plan
        self._document = None
        self._topic.setText(plan.topic)
        self._history.select(plan.report_id)
        self._failure.setVisible(False)
        self._scoping.setVisible(False)
        self._outline.show_plan(plan)
        self._outline.setVisible(True)
        self._stack.setCurrentIndex(_COMPOSE)

    def start_run(self, plan: ReportPlanDraft) -> None:
        """Switches to the running pane. Called once the bridge has accepted the request."""
        self._running_report_id = plan.report_id
        self._steps_done = 0
        self._progress.setRange(0, max(1, plan.expected_job_count))
        self._progress.setValue(0)
        tl.text(self._running_topic, "Researching: {topic}", topic=plan.title or plan.topic)
        tl.text(self._phase, "Starting…")
        self._stop.setEnabled(True)
        tl.text(self._stop, "Stop")
        self._stack.setCurrentIndex(_RUNNING)

    def apply_tick(self, tick: ProgressTick) -> None:
        """
        Advances the bar on report phases only.

        Every job in the process reports through one listener, so a tick from an unrelated phase would
        otherwise push this bar along for work that has nothing to do with the report.
        """
        if self._stack.currentIndex() != _RUNNING or tick.phase not in _REPORT_PHASES:
            return

        if tick.state.value == "completed":
            self._steps_done += 1
            self._progress.setValue(min(self._steps_done, self._progress.maximum()))
        # A general step, never the internal job or its own detail line
        if report_step(tick.phase, tick.detail):
            tl.text(self._phase, lambda phase=tick.phase, detail=tick.detail: report_step(phase, detail))

    def show_report(self, report: ReportDocument) -> None:
        self._document = report
        self._running_report_id = None
        self._failure.setVisible(False)
        self._body.show_report(report)

        tl.text(self._summary, lambda: tr_n(report.section_count, "{n} section", "{n} sections") + ", "
                + tr_n(len(report.evidence), "{n} cited excerpt.", "{n} cited excerpts."))

        if report.is_partial:
            tl.text(self._partial, "This report was stopped before it finished. The sections below are complete "
                                   "and cited; the rest were never written.")
            self._partial.setProperty("role", "banner-warning")
            self._partial.style().unpolish(self._partial)
            self._partial.style().polish(self._partial)
            self._partial.setVisible(True)
        else:
            self._partial.setVisible(False)

        self._sync_resume()
        self._history.select(report.report_id)
        self._stack.setCurrentIndex(_REPORT)
        if report.evidence:
            # Show the first citation straight away: the point is to verify, not to hunt.
            self._source_panel.show_evidence(report.evidence[0])

    def show_failure(self, message: str) -> None:
        self._scoping.setVisible(False)
        tl.text(self._failure, tl.plain(message))   # already in the display language
        self._failure.setProperty("role", "banner-danger")
        self._failure.style().unpolish(self._failure)
        self._failure.style().polish(self._failure)
        self._failure.setVisible(True)
        if self._stack.currentIndex() == _RUNNING:
            self._running_report_id = None
            self._stack.setCurrentIndex(_COMPOSE)

    # ---------- events ----------

    def _selected_mode(self) -> ReportMode:
        """
        QComboBox stores userData as a QVariant, and a str-derived Enum comes back out as a plain str.
        Downstream code compares modes with `is`, which would then silently pick the wrong branch, so
        the enum is rebuilt here at the boundary.
        """
        return ReportMode(self._mode.currentData())


    def _on_plan(self) -> None:
        topic = self._topic.text().strip()
        if not topic:
            return
        self._failure.setVisible(False)
        self.planRequested.emit(topic, self._selected_mode())

    def _on_start(self, plan: ReportPlanDraft) -> None:
        self._plan = plan
        self.reportRequested.emit(plan, self._selected_mode())

    def _on_stop(self) -> None:
        if self._running_report_id is None:
            return
        self._stop.setEnabled(False)
        tl.text(self._stop, "Stopping after this section…")
        self.cancelRequested.emit(self._running_report_id)

    def _on_resume(self) -> None:
        if self._document is not None:
            self.resumeRequested.emit(self._document.report_id)

    def _on_history_chosen(self, report_id: uuid.UUID) -> None:
        ref = self._refs.get(report_id)
        if ref is None:
            return
        if self._stack.currentIndex() == _RUNNING:
            # The run on screen keeps its pane; its progress and Stop button must stay reachable.
            if self._running_report_id is not None:
                self._history.select(self._running_report_id)
            return
        if ref.awaiting_approval:
            self.savedPlanRequested.emit(report_id)
        elif ref.unfinished and not ref.resumable:
            self._on_discard()
            self._topic.setText(ref.topic)
            self._history.select(report_id)
            self._show_note(tr(
                "This report was stopped before it gathered anything, so there is nothing to resume. "
                "Its topic is filled in above: plan it again to start over."
            ))
        else:
            self.savedReportRequested.emit(report_id)

    def _sync_resume(self) -> None:
        document = self._document
        ref = self._refs.get(document.report_id) if document is not None else None
        resumable = document is not None and document.is_partial and (
            ref.resumable if ref is not None else bool(document.evidence))
        self._resume.setVisible(resumable)

    def _show_note(self, message: str) -> None:
        tl.text(self._failure, tl.plain(message))
        self._failure.setProperty("role", "banner-warning")
        self._failure.style().unpolish(self._failure)
        self._failure.style().polish(self._failure)
        self._failure.setVisible(True)

    def _on_discard(self) -> None:
        self._plan = None
        self._document = None
        self._resume.setVisible(False)
        self._history.clear_selection()
        self._outline.setVisible(False)
        self._scoping.setVisible(False)
        self._body.clear_report()
        self._source_panel.clear()
        self._failure.setVisible(False)
        self._stack.setCurrentIndex(_COMPOSE)

    def _on_citation(self, evidence: Evidence) -> None:
        self._source_panel.show_evidence(evidence)

    def _on_citation_missing(self, doc_number: int) -> None:
        """A citation that cannot be checked has to say so, not do nothing."""
        self.show_failure(tr("[Doc {number}] cannot be opened: its source is no longer available.", number=doc_number))

    def _on_save(self) -> None:
        if self._document is None:
            return

        destination, _ = QFileDialog.getSaveFileName(
            self, tr("Save report"), default_save_path(self._document),
            f"{tr('Markdown')} (*.md);;{tr('All files')} (*)"
        )
        if not destination:
            return

        error = save_markdown(self._document, destination)
        if error:
            self.show_failure(error)
        else:
            tl.text(self._summary, "Saved to {path}", path=destination)
