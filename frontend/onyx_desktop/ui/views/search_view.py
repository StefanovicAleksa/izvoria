# Canvas: desktop_search_view_v1
# Path: frontend/onyx_desktop/ui/views/search_view.py

"""
Retrieval without generation: see what the index actually returns for a query.

This is the view for checking the material rather than asking it questions -- when an answer looks
wrong, the useful question is whether the right passages are found at all. Results are shown in rank
order; their scores are not shown (see "What the screen does not say" in frontend/README.md).
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

from onyx_desktop.domain.entities import Evidence, SearchOutcome, SearchRecordRef
from onyx_desktop.domain.enums import SearchMode
from onyx_desktop.ui import translate as tl
from onyx_desktop.ui.widgets.answer_language import AnswerLanguageLabel
from onyx_desktop.ui.widgets.evidence_card import EvidenceCard
from onyx_desktop.ui.widgets.history_list import HistoryEntry, HistoryList
from onyx_desktop.ui.widgets.source_panel import SourcePanel
from onyx_desktop.utils.formatting import duration, when
from onyx_desktop.utils.i18n import N_, tr, tr_n

_MODE_LABELS = {SearchMode.STANDARD: N_("Standard"), SearchMode.VERIFIED: N_("Verified")}


class SearchView(QWidget):
    searchRequested = Signal(str, object, int)   # prompt, SearchMode, limit
    savedSearchChosen = Signal(object)           # record_id
    savedSearchDeleteRequested = Signal(object)  # record_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._cards: List[EvidenceCard] = []
        self._shown: Optional[SearchOutcome] = None
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._history = HistoryList(
            N_("Past searches"), N_("No searches yet. What you search for is kept here, with what it found."),
            deletable=True, delete_label=N_("Delete search"), delete_title=N_("Delete search"),
        )
        self._records: List[SearchRecordRef] = []
        self._history.itemChosen.connect(self.savedSearchChosen)
        self._history.deleteRequested.connect(self.savedSearchDeleteRequested)
        splitter.addWidget(self._history)

        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(24, 20, 16, 20)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.addWidget(tl.label("Search", role="heading"))
        # Verified search's relevance filter reads the question in this language
        self._answer_language = AnswerLanguageLabel()
        header.addWidget(self._answer_language)
        header.addStretch(1)
        layout.addLayout(header)

        blurb = tl.label("Find passages in your material without generating an answer.", role="subtle")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self._input = QLineEdit()
        tl.placeholder(self._input, "Search your material…")
        self._input.returnPressed.connect(self._on_search)
        controls.addWidget(self._input, stretch=1)

        self._mode = QComboBox()
        self._mode.addItem("", SearchMode.STANDARD)
        self._mode.addItem("", SearchMode.VERIFIED)
        tl.items(self._mode, _MODE_LABELS.values())
        tl.tooltip(self._mode, "Verified keeps only the passages that directly answer the query. Slower.")
        controls.addWidget(self._mode)

        self._limit = QSpinBox()
        self._limit.setRange(1, 20)          # SearchQuery caps limit at 20
        self._limit.setValue(5)
        controls.addWidget(self._limit)

        self._button = QPushButton()
        tl.text(self._button, "Search")
        self._button.setProperty("role", "primary")
        self._button.clicked.connect(self._on_search)
        controls.addWidget(self._button)

        layout.addLayout(controls)

        saved_row = QHBoxLayout()
        self._saved_banner = QLabel("")
        self._saved_banner.setProperty("role", "banner-warning")
        self._saved_banner.setWordWrap(True)
        saved_row.addWidget(self._saved_banner, stretch=1)
        self._run_again = QPushButton()
        tl.text(self._run_again, "Run again")
        tl.tooltip(self._run_again, "Search again with the same words and mode, against the material as it is now.")
        self._run_again.clicked.connect(self._on_run_again)
        saved_row.addWidget(self._run_again)
        self._saved_row = QWidget()
        self._saved_row.setLayout(saved_row)
        self._saved_row.hide()
        layout.addWidget(self._saved_row)

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
        right_layout.setContentsMargins(16, 20, 24, 20)
        right_layout.setSpacing(10)

        right_layout.addWidget(tl.label("Source review", role="heading"))

        self._source_panel = SourcePanel()
        right_layout.addWidget(self._source_panel, stretch=1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([230, 620, 420])
        outer.addWidget(splitter)

    # ---------- public API ----------

    def set_answer_language(self, name: str) -> None:
        self._answer_language.set_language(name)

    def set_busy(self, busy: bool) -> None:
        self._button.setEnabled(not busy)
        self._input.setEnabled(not busy)
        self._run_again.setEnabled(not busy)
        tl.text(self._button, "Searching…" if busy else "Search")

    def set_saved_searches(self, records: List[SearchRecordRef]) -> None:
        self._records = list(records)
        tl.on_retranslate(self, "history", self._render_history)

    def _render_history(self) -> None:
        self._history.set_entries([
            HistoryEntry(
                key=record.record_id,
                title=record.query,
                badge=tr(_MODE_LABELS.get(record.mode, record.mode.value.capitalize())),
                subtitle=f"{tr_n(record.hit_count, '{n} hit', '{n} hits')} · {when(record.created_at)}",
            )
            for record in self._records
        ])

    def set_history_unavailable(self, text: str) -> None:
        self._history.set_unavailable(text)

    def show_results(self, outcome: SearchOutcome) -> None:
        self._clear()
        self._shown = outcome
        if outcome.is_saved:
            def note() -> str:
                text = tr("Saved results from {when}: what this search returned then.", when=when(outcome.saved_at))
                if outcome.missing_hits:
                    text += " " + tr_n(outcome.missing_hits, "{n} of these results is no longer available.",
                                       "{n} of these results are no longer available.")
                return text
            tl.text(self._saved_banner, note)
            self._saved_row.show()
            self._input.setText(outcome.query)
            if outcome.mode is not None:
                self._mode.setCurrentIndex(self._mode.findData(outcome.mode))
        else:
            self._saved_row.hide()
            self._history.clear_selection()
        tl.text(self._summary, lambda: tr_n(
            len(outcome.hits), "{n} result for “{query}” in {time}", "{n} results for “{query}” in {time}",
            query=outcome.query, time=duration(outcome.execution_time_ms / 1000.0),
        ))
        if not outcome.hits:
            empty = tl.label("Nothing matched. The material may not have been added yet, or try different wording.")
            empty.setWordWrap(True)
            empty.setProperty("role", "subtle")
            self._insert(empty)
            self._source_panel.clear()
            return

        for hit in outcome.hits:
            card = EvidenceCard(hit.as_evidence())
            card.reviewRequested.connect(self._on_review)
            card.openRequested.connect(self._on_open)
            self._cards.append(card)
            self._insert(card)

    def show_failure(self, message: str) -> None:
        self._clear()
        self._saved_row.hide()
        tl.text(self._summary, tl.plain(""))
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

    def _on_run_again(self) -> None:
        if self._shown is None:
            return
        mode = self._shown.mode or SearchMode(self._mode.currentData())
        self.searchRequested.emit(self._shown.query, mode, self._limit.value())

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
