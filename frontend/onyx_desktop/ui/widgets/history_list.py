# Path: frontend/onyx_desktop/ui/widgets/history_list.py

"""
The history pane every view shares: past conversations, searches or reports, newest first.

It only lists and emits. What reopening an entry means (re-render a conversation, show saved hits,
resume a report) belongs to the view that owns it.
"""

from dataclasses import dataclass
from typing import Hashable, List, Optional

from PySide6.QtCore import QEvent, QRect, QSize, Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

_KEY_ROLE = Qt.ItemDataRole.UserRole
# Matches the stylesheet's item padding (7px 8px), plus a little slack so a descender is never clipped
_ITEM_PADDING_X = 16
_ITEM_PADDING_Y = 9


@dataclass(frozen=True)
class HistoryEntry:
    key: Hashable
    title: str
    subtitle: str = ""
    badge: str = ""


class HistoryList(QWidget):
    itemChosen = Signal(object)       # key
    deleteRequested = Signal(object)  # key

    def __init__(self, title: str, empty_text: str, deletable: bool = False,
                 noun: str = "entry", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._empty_text = empty_text
        self._noun = noun

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 8, 20)
        layout.setSpacing(8)

        heading = QLabel(title)
        heading.setProperty("role", "heading")
        layout.addWidget(heading)

        self._placeholder = QLabel("Start the appliance to see history.")
        self._placeholder.setProperty("role", "subtle")
        self._placeholder.setWordWrap(True)
        # Takes the list's space while the list is hidden, so the heading stays at the top
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._placeholder, stretch=1)

        self._list = QListWidget()
        self._list.setWordWrap(True)
        # Titles wrap, so a horizontal scrollbar only ever shows a sliver of nothing
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideNone)
        # Qt's delegate sizes a wrapped item without the stylesheet's padding and before the list has its
        # width, cutting entries off after the title; each item is sized from its own wrapped text instead.
        self._list.viewport().installEventFilter(self)
        self._list.setUniformItemSizes(False)
        # Two-line entries with wrapped titles run into each other without a gap.
        self._list.setSpacing(5)
        # Clicks only: itemActivated also fires on a double-click, which would load the entry twice.
        self._list.itemClicked.connect(self._on_activated)
        self._list.currentItemChanged.connect(lambda *_: self._sync_buttons())
        self._list.hide()
        layout.addWidget(self._list, stretch=1)

        self._delete = QPushButton(f"Delete {noun}")
        self._delete.setVisible(deletable)
        self._delete.setEnabled(False)
        self._delete.clicked.connect(self._on_delete)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._delete)
        layout.addLayout(row)

    # ---------- public API ----------

    def set_entries(self, entries: List[HistoryEntry]) -> None:
        current = self.current_key()
        self._list.clear()
        for entry in entries:
            lines = [entry.title]
            detail = " · ".join(part for part in (entry.badge, entry.subtitle) if part)
            if detail:
                lines.append(detail)
            item = QListWidgetItem("\n".join(lines))
            item.setData(_KEY_ROLE, entry.key)
            item.setToolTip(entry.title)
            self._list.addItem(item)
            if entry.key == current:
                self._list.setCurrentItem(item)
        self._placeholder.setText(self._empty_text)
        self._placeholder.setVisible(not entries)
        self._list.setVisible(bool(entries))
        self._fit_items()
        self._sync_buttons()

    def eventFilter(self, watched, event) -> bool:
        if watched is self._list.viewport() and event.type() == QEvent.Type.Resize:
            self._fit_items()
        return super().eventFilter(watched, event)

    def _fit_items(self) -> None:
        width = self._list.viewport().width() - 2 * _ITEM_PADDING_X
        if width <= 0:
            return
        metrics = QFontMetrics(self._list.font())
        for index in range(self._list.count()):
            item = self._list.item(index)
            text = metrics.boundingRect(
                QRect(0, 0, width, 100000), int(Qt.TextFlag.TextWordWrap), item.text()
            )
            item.setSizeHint(QSize(width, text.height() + 2 * _ITEM_PADDING_Y))

    def set_unavailable(self, text: str) -> None:
        self._list.clear()
        self._list.hide()
        self._placeholder.setText(text)
        self._placeholder.show()
        self._sync_buttons()

    def select(self, key: Hashable) -> None:
        """Highlights an entry without emitting, e.g. the conversation the view is already showing."""
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item.data(_KEY_ROLE) == key:
                self._list.blockSignals(True)
                self._list.setCurrentItem(item)
                self._list.blockSignals(False)
                break
        self._sync_buttons()

    def clear_selection(self) -> None:
        self._list.blockSignals(True)
        self._list.setCurrentItem(None)
        self._list.clearSelection()
        self._list.blockSignals(False)
        self._sync_buttons()

    def current_key(self) -> Optional[Hashable]:
        item = self._list.currentItem()
        return item.data(_KEY_ROLE) if item is not None else None

    def keys(self) -> List[Hashable]:
        return [self._list.item(i).data(_KEY_ROLE) for i in range(self._list.count())]

    # ---------- events ----------

    def _on_activated(self, item: QListWidgetItem) -> None:
        self._sync_buttons()
        self.itemChosen.emit(item.data(_KEY_ROLE))

    def _on_delete(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        title = item.text().split("\n", 1)[0]
        answer = QMessageBox.question(
            self, f"Delete {self._noun}",
            f"Delete “{title}” from the history? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.deleteRequested.emit(item.data(_KEY_ROLE))

    def _sync_buttons(self) -> None:
        self._delete.setEnabled(self._list.currentItem() is not None)
