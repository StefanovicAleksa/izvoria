# Canvas: desktop_stage_list_v1
# Path: frontend/izvoria_desktop/ui/widgets/stage_list.py

"""
A batch's progress as a short list of general steps, each with its own bar.

It deliberately knows nothing about the engine's jobs: the view feeds it a step key and a fraction, and
`utils/progress_stages.py` decides which job belongs to which step.
"""

from enum import Enum
from typing import Dict, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QProgressBar, QWidget

from izvoria_desktop.utils.progress_stages import Stage
from izvoria_desktop.ui import translate as tl
from izvoria_desktop.utils.i18n import N_

_BAR_STEPS = 1000


class StageState(str, Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"


_MARKS = {
    StageState.WAITING: "○",
    StageState.ACTIVE: "●",
    StageState.DONE: "✓",
    StageState.FAILED: "✕",
}
_CAPTIONS = {
    StageState.WAITING: N_("Waiting"),
    StageState.ACTIVE: N_("In progress"),
    StageState.DONE: N_("Done"),
    StageState.FAILED: N_("Stopped"),
}


class StageList(QFrame):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("role", "card")
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(18, 16, 18, 16)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(12)
        self._grid.setColumnStretch(2, 1)
        self._rows: Dict[str, dict] = {}
        self._order: list = []

    # ---------- public API ----------

    def set_stages(self, stages: Sequence[Stage]) -> None:
        """Lays out a fresh run: every step waiting, every bar empty."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows.clear()
        self._order = [stage.key for stage in stages]

        for row, stage in enumerate(stages):
            mark = QLabel()
            mark.setFixedWidth(16)
            mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label = tl.label(stage.label)
            label.setMinimumWidth(170)
            bar = QProgressBar()
            bar.setObjectName("StageBar")
            bar.setRange(0, _BAR_STEPS)
            bar.setTextVisible(False)
            caption = QLabel()
            caption.setProperty("role", "subtle")
            caption.setMinimumWidth(80)
            caption.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            self._grid.addWidget(mark, row, 0)
            self._grid.addWidget(label, row, 1)
            self._grid.addWidget(bar, row, 2)
            self._grid.addWidget(caption, row, 3)
            self._rows[stage.key] = {"mark": mark, "label": label, "bar": bar, "caption": caption,
                                     "state": None}
            self._set_state(stage.key, StageState.WAITING)

    def advance(self, key: str, fraction: Optional[float]) -> None:
        """
        Makes `key` the running step. Every step before it is done, since the pipeline only moves forward --
        a step with nothing to do for this batch (no visuals in a recording, say) finishes without a tick.
        `fraction` None means the step's size is not known yet, which shows a busy bar.
        """
        if key not in self._rows:
            return
        position = self._order.index(key)
        for earlier in self._order[:position]:
            if self._rows[earlier]["state"] is not StageState.DONE:
                self._complete(earlier)

        self._set_state(key, StageState.ACTIVE)
        bar = self._rows[key]["bar"]
        if fraction is None:
            bar.setRange(0, 0)
        else:
            bar.setRange(0, _BAR_STEPS)
            # Never show a running step as full: the estimate can run ahead of the real work
            bar.setValue(int(min(max(fraction, 0.0), 0.97) * _BAR_STEPS))

    def finish(self) -> None:
        for key in self._order:
            self._complete(key)

    def fail(self) -> None:
        """Marks the running step as where the batch stopped; the ones after it stay waiting."""
        for key in self._order:
            if self._rows[key]["state"] is StageState.ACTIVE:
                self._set_state(key, StageState.FAILED)
                bar = self._rows[key]["bar"]
                bar.setRange(0, _BAR_STEPS)

    def state_of(self, key: str) -> Optional[StageState]:
        row = self._rows.get(key)
        return row["state"] if row else None

    def value_of(self, key: str) -> int:
        return self._rows[key]["bar"].value()

    @property
    def keys(self) -> list:
        return list(self._order)

    # ---------- internals ----------

    def _complete(self, key: str) -> None:
        bar = self._rows[key]["bar"]
        bar.setRange(0, _BAR_STEPS)
        bar.setValue(_BAR_STEPS)
        self._set_state(key, StageState.DONE)

    def _set_state(self, key: str, state: StageState) -> None:
        row = self._rows[key]
        row["state"] = state
        row["mark"].setText(_MARKS[state])
        tl.text(row["caption"], _CAPTIONS[state])
        row["label"].setProperty("role", "strong" if state is StageState.ACTIVE else "")
        for widget in (row["label"],):
            widget.style().unpolish(widget)
            widget.style().polish(widget)
