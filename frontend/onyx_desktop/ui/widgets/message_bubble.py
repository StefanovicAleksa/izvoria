# Canvas: desktop_message_bubble_v1
# Path: frontend/onyx_desktop/ui/widgets/message_bubble.py

"""
One conversation turn.

This widget is where the product's central rule is enforced visually: an answer that came back
with no evidence is marked as untraceable rather than rendered as an ordinary reply, and a
self-critique that was left unresolved is stated rather than buried.
"""

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from onyx_desktop.domain.entities import ChatMessage
from onyx_desktop.domain.enums import MessageRole, TurnOutcome
from onyx_desktop.ui import theme
from onyx_desktop.ui.widgets.evidence_card import EvidenceCard
from onyx_desktop.ui.widgets.reasoning_trace import ReasoningTrace
from onyx_desktop.utils.formatting import duration, split_doc_tags


class MessageBubble(QFrame):
    """Renders a ChatMessage, its evidence cards and (for Reasoned Answer) its reasoning trace."""

    openRequested = Signal(object)     # Evidence
    reviewRequested = Signal(object)   # Evidence

    def __init__(self, message: ChatMessage, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._message = message
        self._cards: List[EvidenceCard] = []
        self.setProperty(
            "role",
            "bubble-user" if message.role is MessageRole.USER else "bubble-assistant",
        )
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        layout.addWidget(self._speaker_label())

        if self._message.is_ungrounded:
            layout.addWidget(self._banner(
                "No source evidence was returned for this answer — it is not traceable to "
                "the ingested material and should not be relied on.",
                "banner-danger",
            ))

        if self._message.outcome is TurnOutcome.NEEDS_CLARIFICATION:
            layout.addWidget(self._banner(
                "The appliance needs a narrower question before it will answer from the "
                "evidence. Reply below and it will re-run with your clarification.",
                "banner-warning",
            ))

        if self._message.critique_unresolved:
            layout.addWidget(self._banner(
                "The answer's own self-critique had unresolved concerns about how well the "
                "evidence supports it. Check the citations closely.",
                "banner-warning",
            ))

        layout.addWidget(self._body())

        if self._message.reasoning:
            layout.addWidget(ReasoningTrace(self._message.reasoning))

        for evidence in self._message.evidence:
            card = EvidenceCard(evidence)
            card.reviewRequested.connect(self.reviewRequested)
            card.openRequested.connect(self.openRequested)
            self._cards.append(card)
            layout.addWidget(card)

        footer = self._footer()
        if footer is not None:
            layout.addWidget(footer)

    def _speaker_label(self) -> QLabel:
        who = "You" if self._message.role is MessageRole.USER else "Onyx"
        label = QLabel(who)
        label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: 11px; font-weight: 600; "
            "letter-spacing: 0.5px;"
        )
        return label

    def _body(self) -> QLabel:
        """
        Citation tags are rendered as coloured chips inside the prose so a claim and its source
        read as one unit. The text was normalized by the gateway, so a plain `[Doc N]` scan is
        reliable here.
        """
        label = QLabel()
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        if self._message.role is MessageRole.USER:
            label.setText(self._message.text)
            return label

        parts = []
        for segment, doc_number in split_doc_tags(self._message.text):
            escaped = (
                segment.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            if doc_number is None:
                parts.append(escaped.replace("\n", "<br>"))
            else:
                parts.append(
                    f'<span style="color:{theme.CITATION}; font-weight:600;">{escaped}</span>'
                )
        label.setText("".join(parts))
        return label

    def _footer(self) -> Optional[QLabel]:
        bits = []
        if self._message.total_iterations:
            bits.append(f"{self._message.total_iterations} reasoning iteration(s)")
        if self._message.latency_ms:
            bits.append(duration(self._message.latency_ms / 1000.0))
        if self._message.evidence:
            bits.append(f"{len(self._message.evidence)} citation(s)")
        if not bits:
            return None
        label = QLabel(" · ".join(bits))
        label.setProperty("role", "subtle")
        return label

    @staticmethod
    def _banner(text: str, role: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setProperty("role", role)
        return label


class MessageRow(QWidget):
    """Aligns a bubble left or right and stops it stretching the full width of the view."""

    openRequested = Signal(object)
    reviewRequested = Signal(object)

    def __init__(self, message: ChatMessage, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        bubble = MessageBubble(message)
        bubble.reviewRequested.connect(self.reviewRequested)
        bubble.openRequested.connect(self.openRequested)

        if message.role is MessageRole.USER:
            layout.addStretch(1)
            layout.addWidget(bubble, stretch=4)
        else:
            layout.addWidget(bubble, stretch=5)
            layout.addStretch(1)
