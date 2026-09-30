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
from pathlib import Path
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
from onyx_desktop.ui import translate as tl
from onyx_desktop.ui.system_open import open_source
from onyx_desktop.ui.widgets.image_thumbnail import set_thumbnail
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

        self._placeholder = tl.label("Select a citation to review the words behind it.")
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
        self._span.setProperty("role", "heading")
        self._span.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self._span)

        self._copy = QPushButton()
        tl.text(self._copy, "⎘ copy")
        tl.tooltip(self._copy, "Copy the cited location")
        self._copy.clicked.connect(self._on_copy)
        header.addWidget(self._copy)

        header.addStretch(1)
        card_layout.addLayout(header)

        self._speakers = QLabel("")
        self._speakers.setProperty("role", "subtle")
        card_layout.addWidget(self._speakers)

        # A cited image's preview: its excerpt is a machine description, so the picture is shown to check it by
        self._thumbnail = QLabel()
        self._thumbnail.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._thumbnail.setVisible(False)
        card_layout.addWidget(self._thumbnail)

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

        self._open = QPushButton()
        tl.text(self._open, "Open source")
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
        tl.text(self._excerpt, tl.plain(evidence.excerpt) if evidence.excerpt
                else "(no text is stored for this excerpt)")

        speakers = speaker_summary(evidence.speakers)
        tl.text(self._speakers, "Speakers: {names}", names=speakers)
        self._speakers.setVisible(bool(speakers))

        # The name only: the full path would lay out the appliance's storage for anyone reading the screen
        self._path.setText(Path(evidence.clip_path).name if evidence.clip_path else "")
        self._path.setVisible(bool(evidence.clip_path))

        if evidence.media_kind is MediaKind.IMAGE:
            set_thumbnail(self._thumbnail, evidence.clip_path, max_side=320)
        else:
            set_thumbnail(self._thumbnail, None)

        span = timespan(evidence.start_sec, evidence.end_sec)
        if evidence.media_kind is MediaKind.IMAGE:
            tl.text(self._kind_note, "This opens the image itself. The excerpt above is an automatic description "
                                     "of it; check it against the picture.")
            tl.text(self._open, "Open image")
            self._open.setEnabled(True)
            tl.tooltip(self._open, tl.plain(""))
        elif evidence.media_kind is MediaKind.DOCUMENT:
            tl.text(self._kind_note, "This opens the whole document. The excerpt above is at {location}.",
                    location=location(evidence))
            tl.text(self._open, "Open document")
            self._open.setEnabled(True)
            tl.tooltip(self._open, "Turn to the page shown above")
        elif evidence.media_kind is MediaKind.CLIP:
            tl.text(self._kind_note, "This file is the cited excerpt itself — it opens at the right moment.")
            tl.text(self._open, "Open clip")
            self._open.setEnabled(True)
            tl.tooltip(self._open, tl.plain(""))
        elif evidence.media_kind is MediaKind.FULL_SOURCE:
            tl.text(self._kind_note, "This opens the whole recording. Skip to {time} to hear it.", time=span)
            tl.text(self._open, "Open full recording")
            self._open.setEnabled(True)
            tl.tooltip(self._open, "Skip to the time shown above")
        else:
            tl.text(self._kind_note, "The source file for this citation is not available. The excerpt above is "
                                     "still the original wording.")
            tl.text(self._open, "Open source")
            self._open.setEnabled(False)
            tl.tooltip(self._open, "The source file is not available")

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
