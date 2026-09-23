# Canvas: desktop_source_panel_v1
# Path: frontend/onyx_desktop/ui/widgets/source_panel.py

"""
The review panel: the verbatim excerpt behind a citation, and a way to open its source.

Playback is deliberately handed off to the operating system's own player rather than embedded.
Onyx is not a media player, and an in-app player meant carrying Qt's whole multimedia stack --
which on this hardware (NVIDIA proprietary driver, no usable VAAPI driver) tried hardware decode
and took the process down with it. Handing the file to the desktop is what actually works, and it
respects whatever player the client already uses.

The consequence to be honest about: a system player cannot be told to start at a timestamp. When
the source is a whole recording rather than an extracted clip, the cited span is therefore shown
prominently and is copyable, so it can be scrubbed to.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from onyx_desktop.domain.entities import Evidence
from onyx_desktop.domain.enums import MediaKind
from onyx_desktop.ui.system_open import open_source
from onyx_desktop.utils.formatting import location, speaker_summary, timespan

logger = logging.getLogger(__name__)


class SourcePanel(QWidget):
    """Shows one Evidence at a time: its excerpt, where it came from, and how to open it."""

    openFailed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._evidence: Optional[Evidence] = None
        self._build()
        self.clear()

    # ---------- construction ----------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._placeholder = QLabel(
            "Select a citation to review the words behind it."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setProperty("role", "subtle")
        layout.addWidget(self._placeholder)

        self._card = QFrame()
        self._card.setProperty("role", "card")
        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(14, 12, 14, 14)
        card_layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)

        self._chip = QLabel("")
        self._chip.setProperty("role", "chip")
        header.addWidget(self._chip)

        # The timestamp is the whole point when the file is a full recording, so it is rendered
        # large rather than as fine print.
        self._span = QLabel("")
        self._span.setStyleSheet("font-size: 17px; font-weight: 600;")
        self._span.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self._span)

        self._copy = QPushButton("⎘ copy")
        self._copy.setToolTip("Copy the cited timestamp range")
        self._copy.clicked.connect(self._on_copy)
        header.addWidget(self._copy)

        header.addStretch(1)
        card_layout.addLayout(header)

        self._speakers = QLabel("")
        self._speakers.setProperty("role", "subtle")
        card_layout.addWidget(self._speakers)

        excerpt_scroll = QScrollArea()
        excerpt_scroll.setWidgetResizable(True)
        excerpt_scroll.setMinimumHeight(140)
        self._excerpt = QLabel("")
        self._excerpt.setWordWrap(True)
        self._excerpt.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._excerpt.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._excerpt.setStyleSheet("font-size: 15px; line-height: 150%;")
        excerpt_scroll.setWidget(self._excerpt)
        card_layout.addWidget(excerpt_scroll, stretch=1)

        self._kind_note = QLabel("")
        self._kind_note.setWordWrap(True)
        self._kind_note.setProperty("role", "subtle")
        card_layout.addWidget(self._kind_note)

        self._open = QPushButton("Open source")
        self._open.setProperty("role", "primary")
        self._open.clicked.connect(self._on_open)
        card_layout.addWidget(self._open)

        self._path = QLabel("")
        self._path.setWordWrap(True)
        self._path.setProperty("role", "subtle")
        self._path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card_layout.addWidget(self._path)

        self._problem = QLabel("")
        self._problem.setWordWrap(True)
        self._problem.setProperty("role", "banner-warning")
        self._problem.setVisible(False)
        card_layout.addWidget(self._problem)

        layout.addWidget(self._card, stretch=1)

    # ---------- public API ----------

    def show_evidence(self, evidence: Evidence) -> None:
        self._evidence = evidence
        self._placeholder.setVisible(False)
        self._card.setVisible(True)
        self._problem.setVisible(False)

        self._chip.setText(evidence.doc_label)
        self._span.setText(location(evidence))
        self._excerpt.setText(
            evidence.excerpt or "(no transcript text is stored for this chunk)"
        )

        speakers = speaker_summary(evidence.speakers)
        self._speakers.setText(f"Speakers: {speakers}" if speakers else "")
        self._speakers.setVisible(bool(speakers))

        self._path.setText(evidence.clip_path or "")
        self._path.setVisible(bool(evidence.clip_path))

        if evidence.media_kind is MediaKind.DOCUMENT:
            self._kind_note.setText(
                f"This opens the whole document. The excerpt above is at {location(evidence)}."
            )
            self._open.setText("Open document")
            self._open.setEnabled(True)
            self._open.setToolTip("A system viewer cannot be told which page to show, so turn to the page above")
        elif evidence.media_kind is MediaKind.CLIP:
            self._kind_note.setText(
                "This file is the cited excerpt itself — it opens at the right moment."
            )
            self._open.setText("Open clip")
            self._open.setEnabled(True)
            self._open.setToolTip("")
        elif evidence.media_kind is MediaKind.FULL_SOURCE:
            self._kind_note.setText(
                f"No clip was cut for this chunk, so this opens the whole recording. "
                f"Skip to {timespan(evidence.start_sec, evidence.end_sec)} to hear it."
            )
            self._open.setText("Open full recording")
            self._open.setEnabled(True)
            self._open.setToolTip(
                "A system player cannot be told where to start, so seek to the timestamp above"
            )
        else:
            self._kind_note.setText(
                "No source file was resolved for this citation, so there is nothing to open. "
                "The excerpt above is still the verbatim transcript."
            )
            self._open.setText("Open source")
            self._open.setEnabled(False)
            self._open.setToolTip("This citation has no file on disk")

    def open_current(self) -> None:
        """Opens whatever is currently shown. Used when a card's own button was the click."""
        self._on_open()

    def clear(self) -> None:
        self._evidence = None
        self._card.setVisible(False)
        self._placeholder.setVisible(True)

    # ---------- events ----------

    def _on_copy(self) -> None:
        if self._evidence is None:
            return
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(location(self._evidence))

    def _on_open(self) -> None:
        if self._evidence is None:
            return
        problem = open_source(self._evidence)
        if problem is not None:
            self._fail(problem)

    def _fail(self, message: str) -> None:
        self._problem.setText(message)
        self._problem.setVisible(True)
        self.openFailed.emit(message)
