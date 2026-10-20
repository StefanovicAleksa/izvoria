# Canvas: desktop_message_bubble_v1
# Path: frontend/izvoria_desktop/ui/widgets/message_bubble.py

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

from izvoria_desktop.domain.entities import ChatMessage
from izvoria_desktop.domain.enums import MessageRole, TurnOutcome
from izvoria_desktop.ui import theme
from izvoria_desktop.ui import translate as tl
from izvoria_desktop.ui.widgets.evidence_card import EvidenceCard
from izvoria_desktop.utils.formatting import duration, split_doc_tags
from izvoria_desktop.utils.i18n import N_, tr_n


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
                N_("No sources were found for this answer — it cannot be traced to your material "
                "and should not be relied on."),
                "banner-danger",
            ))

        if self._message.outcome is TurnOutcome.NEEDS_CLARIFICATION:
            layout.addWidget(self._banner(
                N_("The appliance needs a narrower question before it can answer from your material. "
                "Reply below and it will try again with your clarification."),
                "banner-warning",
            ))

        if self._message.critique_unresolved:
            layout.addWidget(self._banner(
                N_("Parts of this answer are only weakly supported by the sources. Check the citations "
                "closely."),
                "banner-warning",
            ))

        if self._message.scoped_source:
            # Which file the question was read as naming: a wrong pick is visible, not silent
            scope = tl.label("Answered from: {title}", role="subtle", title=self._message.scoped_source)
            scope.setWordWrap(True)
            layout.addWidget(scope)

        layout.addWidget(self._body())

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
        if self._message.role is MessageRole.USER:
            return tl.label("You", role="speaker")
        return tl.label(tl.plain("Izvoria"), role="speaker")   # a name, the same in every language

    def _body(self) -> QLabel:
        """
        Citation tags are rendered as coloured chips inside the prose so a claim and its source
        read as one unit. The text was normalized by the gateway, so a plain `[Doc N]` scan is
        reliable here.
        """
        label = QLabel()
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._body_label = label
        self._render_body()
        return label

    def apply_theme(self) -> None:
        """The citation colour is inside the rich text, out of the stylesheet's reach, so it is re-rendered."""
        self._render_body()

    def _render_body(self) -> None:
        label = self._body_label
        if self._message.role is MessageRole.USER:
            label.setText(self._message.text)   # the user's own words: never translated
            return

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

    def _footer(self) -> Optional[QLabel]:
        message = self._message
        if not message.latency_ms and not message.evidence:
            return None

        def footer() -> str:
            bits = []
            if message.latency_ms:
                bits.append(duration(message.latency_ms / 1000.0))
            if message.evidence:
                bits.append(tr_n(len(message.evidence), "{n} citation", "{n} citations"))
            return " · ".join(bits)

        return tl.label(footer, role="subtle")

    @staticmethod
    def _banner(text: str, role: str) -> QLabel:
        label = tl.label(text, role=role)
        label.setWordWrap(True)
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
