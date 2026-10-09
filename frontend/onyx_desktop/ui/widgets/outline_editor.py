# Canvas: desktop_outline_editor_v1
# Path: frontend/onyx_desktop/ui/widgets/outline_editor.py

"""
The approval step: the plan the appliance proposes, before it spends half an hour on it.

The point of this widget is not document editing -- it is catching a misread topic while that is
still cheap. So it does the few things that actually change the outcome: rename a section, fix what
it is meant to cover, drop one that is off-target, and reword a sub-question that will not find
anything.

Section keys are carried through untouched. The engine re-attaches evidence by key across plan
revisions, so a renamed or reordered section keeps its identity; regenerating keys here would
silently detach them.
"""

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QTextEdit, QVBoxLayout, QWidget,
)

from onyx_desktop.domain.entities import ReportOutlineDraft, ReportPlanDraft
from onyx_desktop.ui import theme
from onyx_desktop.ui import translate as tl
from onyx_desktop.utils.i18n import tr


class _SectionCard(QFrame):
    """One editable section: title, brief, its sub-questions, and a way to drop it."""

    removeRequested = Signal(str)   # section key

    def __init__(self, section: ReportOutlineDraft, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._key = section.key
        self.setProperty("role", "card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self._title = QLineEdit(section.title)
        tl.placeholder(self._title, "Section heading")
        header.addWidget(self._title, stretch=1)

        remove = QPushButton()
        tl.text(remove, "Drop")
        tl.tooltip(remove, "Leave this section out of the report entirely.")
        remove.clicked.connect(lambda: self.removeRequested.emit(self._key))
        header.addWidget(remove)
        layout.addLayout(header)

        self._brief = QTextEdit(section.brief)
        tl.placeholder(self._brief, "What this section must cover")
        self._brief.setFixedHeight(54)
        layout.addWidget(self._brief)

        questions = tl.label("What to look for", role="subtle")
        layout.addWidget(questions)

        self._questions = QTextEdit("\n".join(section.questions))
        tl.placeholder(self._questions, "One topic or phrase per line")
        self._questions.setFixedHeight(56)
        tl.tooltip(self._questions, "Specific wording that appears in your material finds more than an abstract theme.")
        layout.addWidget(self._questions)

    @property
    def key(self) -> str:
        return self._key

    def to_draft(self) -> ReportOutlineDraft:
        return ReportOutlineDraft(
            key=self._key,
            title=self._title.text().strip(),
            brief=self._brief.toPlainText().strip(),
            questions=tuple(
                line.strip() for line in self._questions.toPlainText().split("\n") if line.strip()
            ),
        )


class OutlineEditor(QWidget):
    """The proposed plan, editable, with Start and Discard."""

    accepted = Signal(object)   # ReportPlanDraft
    discarded = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._plan: Optional[ReportPlanDraft] = None
        self._cards: List[_SectionCard] = []
        self._build()

    # ---------- construction ----------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(tl.label("Report title", role="strong"))
        # Proposed by the scoping model; what the report, its file and the history list are called
        self._title = QLineEdit()
        self._title.setObjectName("ReportTitle")
        self._title.setMaxLength(120)
        layout.addWidget(self._title)

        self._request = QLabel()
        self._request.setWordWrap(True)
        self._request.setProperty("role", "subtle")
        layout.addWidget(self._request)

        self._scope = QLabel()
        self._scope.setWordWrap(True)
        self._scope.setProperty("role", "subtle")
        layout.addWidget(self._scope)

        self._cost = QLabel()
        self._cost.setProperty("role", "subtle")
        layout.addWidget(self._cost)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._sections = QWidget()
        self._sections_layout = QVBoxLayout(self._sections)
        self._sections_layout.setContentsMargins(0, 0, 0, 0)
        self._sections_layout.setSpacing(8)
        self._sections_layout.addStretch(1)
        scroll.setWidget(self._sections)
        layout.addWidget(scroll, stretch=1)

        self._warning = QLabel()
        self._warning.setWordWrap(True)
        self._warning.setVisible(False)
        layout.addWidget(self._warning)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        discard = QPushButton()
        tl.text(discard, "Discard")
        discard.clicked.connect(self.discarded.emit)
        buttons.addWidget(discard)

        self._start = QPushButton()
        tl.text(self._start, "Start research")
        self._start.setProperty("role", "primary")
        self._start.clicked.connect(self._on_start)
        buttons.addWidget(self._start)
        layout.addLayout(buttons)

    # ---------- public API ----------

    def show_plan(self, plan: ReportPlanDraft) -> None:
        self._plan = plan
        self._clear_cards()
        self._warning.setVisible(False)

        self._title.setText(plan.title)
        tl.placeholder(self._title, tl.plain(plan.topic))
        tl.text(self._request, "Request: {topic}", topic=plan.topic)

        # The scope note is written by the engine, in the report's language: shown as it is
        tl.text(self._scope, tl.plain(plan.scope_note) if plan.scope_note
                else "The appliance proposed this outline from your material.")
        tl.text(self._cost, "Roughly {minutes} minutes; chat and search are unavailable while it runs.",
                minutes=plan.estimated_minutes)

        for section in plan.sections:
            card = _SectionCard(section)
            card.removeRequested.connect(self._on_remove)
            self._cards.append(card)
            self._sections_layout.insertWidget(self._sections_layout.count() - 1, card)

    def set_busy(self, busy: bool) -> None:
        self._start.setEnabled(not busy)

    # ---------- events ----------

    def _on_remove(self, key: str) -> None:
        if len(self._cards) == 1:
            self._warn(tr("A report needs at least one section."))
            return
        for card in list(self._cards):
            if card.key == key:
                self._cards.remove(card)
                card.setParent(None)
                card.deleteLater()

    def _on_start(self) -> None:
        if self._plan is None:
            return

        sections = [card.to_draft() for card in self._cards]
        untitled = [index + 1 for index, s in enumerate(sections) if not s.title]
        if untitled:
            self._warn(tr("Section {number} needs a heading before this can run.", number=untitled[0]))
            return
        if not sections:
            self._warn(tr("A report needs at least one section."))
            return

        self._warning.setVisible(False)
        self.accepted.emit(
            ReportPlanDraft(
                report_id=self._plan.report_id,
                topic=self._plan.topic,
                title=" ".join(self._title.text().split()),
                scope_note=self._plan.scope_note,
                sections=tuple(sections),
                indexed_source_count=self._plan.indexed_source_count,
                indexed_chunk_count=self._plan.indexed_chunk_count,
                expected_job_count=self._plan.expected_job_count,
                estimated_minutes=self._plan.estimated_minutes,
            )
        )

    # ---------- plumbing ----------

    def _warn(self, message: str) -> None:
        tl.text(self._warning, tl.plain(message))
        self._warning.setProperty("role", "banner-warning")
        self._warning.style().unpolish(self._warning)
        self._warning.style().polish(self._warning)
        self._warning.setVisible(True)

    def _clear_cards(self) -> None:
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards = []
