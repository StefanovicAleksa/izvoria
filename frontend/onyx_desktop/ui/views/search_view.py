# Canvas: desktop_search_view_v1
# Path: frontend/onyx_desktop/ui/views/search_view.py

"""
Retrieval without generation: see what the index actually returns for a query.

This is the view for diagnosing the appliance rather than asking it questions -- when an answer
looks wrong, the useful question is whether retrieval found the right material at all. Scores are
shown here (and only here) because these results are genuinely ranked.
"""

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from onyx_desktop.domain.entities import Evidence, SearchOutcome
from onyx_desktop.domain.enums import SearchMode
from onyx_desktop.ui.widgets.evidence_card import EvidenceCard
from onyx_desktop.ui.widgets.source_panel import SourcePanel
from onyx_desktop.utils.formatting import duration


class SearchView(QWidget):
    searchRequested = Signal(str, object, int)   # prompt, SearchMode, limit

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._cards: List[EvidenceCard] = []
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(16, 14, 12, 14)
        layout.setSpacing(10)

        title = QLabel("Search")
        title.setProperty("role", "heading")
        layout.addWidget(title)

        blurb = QLabel(
            "Raw retrieval, with no answer generated on top. Use it to check what the index "
            "returns for a query."
        )
        blurb.setProperty("role", "subtle")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Search the ingested material…")
        self._input.returnPressed.connect(self._on_search)
        controls.addWidget(self._input, stretch=1)

        self._mode = QComboBox()
        self._mode.addItem("Standard", SearchMode.STANDARD)
        self._mode.addItem("Verified", SearchMode.VERIFIED)
        self._mode.setToolTip(
            "Verified adds an SLM relevance pass that drops chunks which do not actually answer "
            "the query. Slower, and it loads a second model."
        )
        controls.addWidget(self._mode)

        self._limit = QSpinBox()
        self._limit.setRange(1, 20)          # SearchQuery caps limit at 20
        self._limit.setValue(5)
        controls.addWidget(self._limit)

        self._button = QPushButton("Search")
        self._button.setProperty("role", "primary")
        self._button.clicked.connect(self._on_search)
        controls.addWidget(self._button)

        layout.addLayout(controls)

        self._summary = QLabel("")
        self._summary.setProperty("role", "subtle")
        layout.addWidget(self._summary)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._results = QWidget()
        self._results_layout = QVBoxLayout(self._results)
        self._results_layout.setContentsMargins(0, 0, 8, 0)
        self._results_layout.setSpacing(10)
        self._results_layout.addStretch(1)
        self._scroll.setWidget(self._results)
        layout.addWidget(self._scroll, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 14, 16, 14)
        right_layout.setSpacing(10)

        review_title = QLabel("Source review")
        review_title.setProperty("role", "heading")
        right_layout.addWidget(review_title)

        self._source_panel = SourcePanel()
        right_layout.addWidget(self._source_panel, stretch=1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter)

    # ---------- public API ----------

    def set_busy(self, busy: bool) -> None:
        self._button.setEnabled(not busy)
        self._input.setEnabled(not busy)
        self._button.setText("Searching…" if busy else "Search")

    def show_results(self, outcome: SearchOutcome) -> None:
        self._clear()
        self._summary.setText(
            f"{len(outcome.hits)} result(s) for “{outcome.query}” "
            f"in {duration(outcome.execution_time_ms / 1000.0)}"
        )
        if not outcome.hits:
            empty = QLabel(
                "Nothing matched. Either the material has not been ingested, or the wording is "
                "far from how it was said on the recording."
            )
            empty.setWordWrap(True)
            empty.setProperty("role", "subtle")
            self._insert(empty)
            self._source_panel.clear()
            return

        for hit in outcome.hits:
            card = EvidenceCard(hit.as_evidence(), show_scores=True)
            card.reviewRequested.connect(self._on_review)
            card.openRequested.connect(self._on_open)
            self._cards.append(card)
            self._insert(card)

    def show_failure(self, message: str) -> None:
        self._clear()
        self._summary.setText("")
        label = QLabel(message)
        label.setWordWrap(True)
        label.setProperty("role", "banner-danger")
        self._insert(label)

    # ---------- events ----------

    def _on_search(self) -> None:
        prompt = self._input.text().strip()
        if not prompt:
            return
        # Rebuilt from the combo's QVariant: a str-derived Enum comes back as a plain str, and an
        # `is` comparison in the gateway would then silently run a standard search instead.
        self.searchRequested.emit(
            prompt, SearchMode(self._mode.currentData()), self._limit.value()
        )

    def _on_review(self, evidence: Evidence) -> None:
        self._source_panel.show_evidence(evidence)

    def _on_open(self, evidence: Evidence) -> None:
        """One click from a card straight to the player; the panel also shows what was opened."""
        self._source_panel.show_evidence(evidence)
        self._source_panel.open_current()

    # ---------- plumbing ----------

    def _insert(self, widget: QWidget) -> None:
        self._results_layout.insertWidget(self._results_layout.count() - 1, widget)

    def _clear(self) -> None:
        self._cards = []
        while self._results_layout.count() > 1:
            item = self._results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
