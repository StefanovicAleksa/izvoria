# Canvas: desktop_evidence_card_v2
# Path: frontend/onyx_desktop/ui/widgets/evidence_card.py

"""
One cited excerpt, rendered so a reader can check a claim against the transcript.

The primary button opens the source in the desktop's own player in **one** click, because that is
what a reader wants when they doubt a claim. Clicking the card itself instead loads it into the
review panel to read the full excerpt without launching anything.

The excerpt text is always the verbatim speaker-labelled transcript. When a card has no file its
button is disabled *with a reason*, never quietly hidden: an operator needs to be able to tell
"this claim has no reviewable source" apart from "I forgot to look".
"""

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

from onyx_desktop.domain.entities import Evidence
from onyx_desktop.domain.enums import MediaKind
from onyx_desktop.ui import theme
from onyx_desktop.utils.formatting import score, speaker_summary, timespan


class EvidenceCard(QFrame):
    """A single piece of evidence."""

    openRequested = Signal(object)     # Evidence -- launch the system player now
    reviewRequested = Signal(object)   # Evidence -- show it in the review panel

    def __init__(self, evidence: Evidence, show_scores: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._evidence = evidence
        self.setProperty("role", "card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build(show_scores)

    @property
    def evidence(self) -> Evidence:
        return self._evidence

    def mousePressEvent(self, event) -> None:
        """Clicking the card reviews it; only the button launches a player."""
        self.reviewRequested.emit(self._evidence)
        super().mousePressEvent(event)

    def _build(self, show_scores: bool) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)

        chip = QLabel(self._evidence.doc_label)
        chip.setProperty("role", "chip")
        header.addWidget(chip)

        span = QLabel(timespan(self._evidence.start_sec, self._evidence.end_sec))
        span.setProperty("role", "subtle")
        header.addWidget(span)

        speakers = speaker_summary(self._evidence.speakers)
        if speakers:
            who = QLabel(f"· {speakers}")
            who.setProperty("role", "subtle")
            header.addWidget(who)

        header.addStretch(1)

        # Scores only mean something on ranked results. A hydrated citation has none, so showing
        # them there would be inventing precision that does not exist.
        if show_scores:
            score_text = f"rrf {score(self._evidence.relevance_score)}"
            if self._evidence.rerank_score is not None:
                score_text += f" · rerank {score(self._evidence.rerank_score)}"
            scores = QLabel(score_text)
            scores.setProperty("role", "subtle")
            header.addWidget(scores)

        layout.addLayout(header)

        excerpt = QLabel(self._evidence.excerpt or "(no transcript text available for this chunk)")
        excerpt.setWordWrap(True)
        excerpt.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if not self._evidence.excerpt:
            excerpt.setProperty("role", "subtle")
        layout.addWidget(excerpt)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(self._open_button())

        if self._evidence.clip_path:
            path = QLabel(self._evidence.clip_path)
            path.setProperty("role", "subtle")
            path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            path.setToolTip(self._evidence.clip_path)
            footer.addWidget(path, stretch=1)
        else:
            missing = QLabel("source unresolved")
            missing.setStyleSheet(f"color: {theme.WARNING};")
            footer.addWidget(missing, stretch=1)

        layout.addLayout(footer)

    def _open_button(self) -> QPushButton:
        """
        One click straight to the player. The label names what will actually open, since a chunk
        without an extracted clip yields the entire recording rather than the excerpt.
        """
        button = QPushButton()
        button.setProperty("role", "primary")

        if not self._evidence.is_playable:
            button.setText("Open source")
            button.setEnabled(False)
            button.setToolTip(
                "No source file was resolved for this citation, so there is nothing to open"
            )
            return button

        button.clicked.connect(lambda: self.openRequested.emit(self._evidence))
        if self._evidence.media_kind is MediaKind.FULL_SOURCE:
            button.setText("Open recording ↗")
            button.setToolTip(
                "No clip was cut for this chunk, so the whole recording opens — skip to "
                f"{timespan(self._evidence.start_sec, self._evidence.end_sec)}"
            )
        else:
            button.setText("Open clip ↗")
            button.setToolTip("Opens this exact excerpt in your video player")
        return button
