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
from onyx_desktop.domain.entities import Evidence, ProgressTick, ReportDocument, ReportPlanDraft
from onyx_desktop.domain.enums import ReportMode
from onyx_desktop.ui import theme
from onyx_desktop.ui.widgets.outline_editor import OutlineEditor
from onyx_desktop.ui.widgets.report_document import ReportDocumentView
from onyx_desktop.ui.widgets.source_panel import SourcePanel

logger = logging.getLogger(__name__)

_COMPOSE, _RUNNING, _REPORT = 0, 1, 2

# Job phases a report emits. Used to ignore ticks from anything else.
_REPORT_PHASES = {
    "research_scoping", "research_outline", "research_section", "research_summary",
    "rag_retrieval", "rag_reranking", "retrieval_scrubbing",
}


class ReportView(QWidget):
    planRequested = Signal(str, object)      # topic, ReportMode
    reportRequested = Signal(object, object)  # ReportPlanDraft, ReportMode
    cancelRequested = Signal(object)          # report_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._plan: Optional[ReportPlanDraft] = None
        self._document: Optional[ReportDocument] = None
        self._steps_done = 0
        self._build()

    # ---------- construction ----------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 14, 12, 14)
        left_layout.setSpacing(10)

        heading = QLabel("Research report")
        heading.setProperty("role", "heading")
        left_layout.addWidget(heading)

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
        right_layout.setContentsMargins(12, 14, 16, 14)
        right_layout.setSpacing(10)
        review = QLabel("Source review")
        review.setProperty("role", "heading")
        right_layout.addWidget(review)
        hint = QLabel("Click a [Doc N] citation in the report to check it against the recording.")
        hint.setWordWrap(True)
        hint.setProperty("role", "subtle")
        right_layout.addWidget(hint)
        self._source_panel = SourcePanel()
        right_layout.addWidget(self._source_panel, stretch=1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    def _compose_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        blurb = QLabel(
            "Give a topic. The appliance searches your archive, proposes an outline for you to "
            "approve, then writes a cited report from what it finds."
        )
        blurb.setWordWrap(True)
        blurb.setProperty("role", "subtle")
        layout.addWidget(blurb)

        self._topic = QLineEdit()
        self._topic.setPlaceholderText("e.g. chain of custody for the evidence bag")
        self._topic.returnPressed.connect(self._on_plan)
        layout.addWidget(self._topic)

        controls = QHBoxLayout()
        self._mode = QComboBox()
        self._mode.addItem("Deep Research — one sweep", ReportMode.DEEP)
        self._mode.addItem("Exhaustive — repeats until nothing new", ReportMode.EXHAUSTIVE)
        self._mode.setToolTip(
            "Exhaustive re-searches for any section the first sweep could not support, using wording "
            "it found in your archive. Slower, but 'not covered' then means the material really is "
            "absent rather than badly asked for."
        )
        controls.addWidget(self._mode, stretch=1)

        self._plan_button = QPushButton("Plan research")
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

        self._phase = QLabel("Starting…")
        self._phase.setWordWrap(True)
        self._phase.setProperty("role", "subtle")
        layout.addWidget(self._phase)

        note = QLabel(
            "Chat and search are unavailable while this runs — the appliance has one GPU and this "
            "run is using it."
        )
        note.setWordWrap(True)
        note.setProperty("role", "subtle")
        layout.addWidget(note)

        self._stop = QPushButton("Stop")
        self._stop.setToolTip(
            "Stops after the section currently being written — up to about a minute and a half. "
            "Everything already written is kept."
        )
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
        new_report = QPushButton("New report")
        new_report.clicked.connect(self._on_discard)
        buttons.addWidget(new_report)
        buttons.addStretch(1)

        self._save = QPushButton("Save as Markdown…")
        self._save.setProperty("role", "primary")
        self._save.clicked.connect(self._on_save)
        buttons.addWidget(self._save)
        layout.addLayout(buttons)
        return pane

    # ---------- public API ----------

    def set_busy(self, busy: bool) -> None:
        self._plan_button.setEnabled(not busy)
        self._topic.setEnabled(not busy)
        self._mode.setEnabled(not busy)
        self._outline.set_busy(busy)

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
        self._scoping.setText(
            f"Searching the archive and drafting an outline for \u201c{topic}\u201d\u2026 "
            f"This takes about a minute."
        )
        self._scoping.setVisible(True)
        self._stack.setCurrentIndex(_COMPOSE)

    def show_plan(self, plan: ReportPlanDraft) -> None:
        self._plan = plan
        self._failure.setVisible(False)
        self._scoping.setVisible(False)
        self._outline.show_plan(plan)
        self._outline.setVisible(True)
        self._stack.setCurrentIndex(_COMPOSE)

    def start_run(self, plan: ReportPlanDraft) -> None:
        """Switches to the running pane. Called once the bridge has accepted the request."""
        self._steps_done = 0
        self._progress.setRange(0, max(1, plan.expected_job_count))
        self._progress.setValue(0)
        self._running_topic.setText(f"Researching: {plan.topic}")
        self._phase.setText("Starting…")
        self._stop.setEnabled(True)
        self._stop.setText("Stop")
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
        if tick.detail:
            self._phase.setText(tick.detail)

    def show_report(self, report: ReportDocument) -> None:
        self._document = report
        self._failure.setVisible(False)
        self._body.show_report(report)

        cited = len(report.evidence)
        self._summary.setText(
            f"{report.section_count} section(s), {cited} cited excerpt(s), "
            f"{report.rounds_used} gathering round(s)."
        )

        if report.is_partial:
            self._partial.setText(
                "This report was stopped before it finished. The sections below are complete and cited; "
                "the rest were never written."
            )
            self._partial.setProperty("role", "banner-warning")
            self._partial.style().unpolish(self._partial)
            self._partial.style().polish(self._partial)
            self._partial.setVisible(True)
        else:
            self._partial.setVisible(False)

        self._stack.setCurrentIndex(_REPORT)
        if report.evidence:
            # Show the first citation straight away: the point is to verify, not to hunt.
            self._source_panel.show_evidence(report.evidence[0])

    def show_failure(self, message: str) -> None:
        self._scoping.setVisible(False)
        self._failure.setText(message)
        self._failure.setProperty("role", "banner-danger")
        self._failure.style().unpolish(self._failure)
        self._failure.style().polish(self._failure)
        self._failure.setVisible(True)
        if self._stack.currentIndex() == _RUNNING:
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
        if self._plan is None:
            return
        self._stop.setEnabled(False)
        self._stop.setText("Stopping after this section…")
        self.cancelRequested.emit(self._plan.report_id)

    def _on_discard(self) -> None:
        self._plan = None
        self._document = None
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
        self.show_failure(
            f"[Doc {doc_number}] cannot be opened: its excerpt is no longer in the evidence bank, "
            f"most likely because that recording was re-ingested."
        )

    def _on_save(self) -> None:
        if self._document is None:
            return

        destination, _ = QFileDialog.getSaveFileName(
            self, "Save report", default_save_path(self._document), "Markdown (*.md);;All files (*)"
        )
        if not destination:
            return

        error = save_markdown(self._document, destination)
        if error:
            self.show_failure(error)
        else:
            self._summary.setText(f"Saved to {destination}")
