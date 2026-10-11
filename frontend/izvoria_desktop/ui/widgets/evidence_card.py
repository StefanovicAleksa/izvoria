# Canvas: desktop_evidence_card_v2
# Path: frontend/izvoria_desktop/ui/widgets/evidence_card.py

"""
One cited excerpt, rendered so a reader can check a claim against the transcript.

The primary button opens the source in the desktop's own player in **one** click, because that is
what a reader wants when they doubt a claim. Clicking the card itself instead loads it into the
review panel to read the full excerpt without launching anything.

The excerpt text is the verbatim speaker-labelled transcript or document text; for an image it is the
labelled machine description, shown beside a thumbnail of the image itself. When a card has no file its
button is disabled *with a reason*, never quietly hidden: an operator needs to be able to tell
"this claim has no reviewable source" apart from "I forgot to look".
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from izvoria_desktop.domain.entities import Evidence
from izvoria_desktop.domain.enums import MediaKind
from izvoria_desktop.ui import translate as tl
from izvoria_desktop.ui.widgets.image_thumbnail import set_thumbnail
from izvoria_desktop.utils.formatting import location, speaker_summary, timespan


class EvidenceCard(QFrame):
    """A single piece of evidence."""

    openRequested = Signal(object)     # Evidence -- launch the system player now
    reviewRequested = Signal(object)   # Evidence -- show it in the review panel

    def __init__(self, evidence: Evidence, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._evidence = evidence
        self.setProperty("role", "card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build()

    @property
    def evidence(self) -> Evidence:
        return self._evidence

    def mousePressEvent(self, event) -> None:
        """Clicking the card reviews it; only the button launches a player."""
        self.reviewRequested.emit(self._evidence)
        super().mousePressEvent(event)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)

        chip = QLabel(self._evidence.doc_label)
        chip.setProperty("role", "chip")
        header.addWidget(chip)

        span = QLabel(location(self._evidence))
        span.setProperty("role", "subtle")
        header.addWidget(span)

        speakers = speaker_summary(self._evidence.speakers)
        if speakers:
            who = QLabel(f"· {speakers}")
            who.setProperty("role", "subtle")
            header.addWidget(who)

        header.addStretch(1)

        layout.addLayout(header)

        excerpt = QLabel()
        tl.text(excerpt, tl.plain(self._evidence.excerpt) if self._evidence.excerpt
                else "(no text available for this excerpt)")
        excerpt.setWordWrap(True)
        excerpt.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if not self._evidence.excerpt:
            excerpt.setProperty("role", "subtle")

        if self._evidence.media_kind is MediaKind.IMAGE:
            body = QHBoxLayout()
            body.setSpacing(12)
            self._thumbnail = QLabel()
            self._thumbnail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            set_thumbnail(self._thumbnail, self._evidence.clip_path)
            body.addWidget(self._thumbnail, alignment=Qt.AlignmentFlag.AlignTop)
            body.addWidget(excerpt, stretch=1)
            layout.addLayout(body)
        else:
            layout.addWidget(excerpt)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(self._open_button())

        if self._evidence.clip_path:
            path = QLabel(Path(self._evidence.clip_path).name)
            path.setProperty("role", "subtle")
            path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            footer.addWidget(path, stretch=1)
        else:
            missing = tl.label("source not available")
            missing.setProperty("role", "warning-text")
            footer.addWidget(missing, stretch=1)

        layout.addLayout(footer)

    def _open_button(self) -> QPushButton:
        """
        One click straight to the player. The label names what will actually open, since an excerpt
        without its own clip opens the entire recording.
        """
        button = QPushButton()
        button.setProperty("role", "primary")

        if not self._evidence.is_playable:
            tl.text(button, "Open source")
            button.setEnabled(False)
            tl.tooltip(button, "The source file for this citation is not available")
            return button

        button.clicked.connect(lambda: self.openRequested.emit(self._evidence))
        if self._evidence.media_kind is MediaKind.IMAGE:
            tl.text(button, "Open image ↗")
            tl.tooltip(button, "Opens the image in your image viewer — the excerpt is an automatic description of it")
        elif self._evidence.media_kind is MediaKind.DOCUMENT:
            tl.text(button, "Open document ↗")
            tl.tooltip(button, "Opens the whole document — the excerpt is at {location}",
                       location=location(self._evidence))
        elif self._evidence.media_kind is MediaKind.FULL_SOURCE:
            tl.text(button, "Open recording ↗")
            tl.tooltip(button, "Opens the whole recording — skip to {time}",
                       time=timespan(self._evidence.start_sec, self._evidence.end_sec))
        else:
            tl.text(button, "Open clip ↗")
            tl.tooltip(button, "Opens this exact excerpt in your video player")
        return button
