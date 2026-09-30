# Canvas: desktop_chat_view_v1
# Path: frontend/onyx_desktop/ui/views/chat_view.py

"""
The conversation view: ask, read the answer, click a citation, check it against the transcript.

Two behaviours here are dictated by the engine rather than by taste:

- Reasoned Answer often answers with a question instead of an answer, and its intra-question state
  is discarded when the call returns, so a clarification cannot resume mid-loop. The reply is
  therefore sent as the original question with `[Clarification N]` lines appended, which is what
  the CLI does. Capped at three rounds.
- A turn takes tens of seconds, so the composer is disabled for the duration and the transcript
  shows a pending placeholder rather than appearing to have swallowed the question.
"""

import uuid
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from onyx_desktop.domain.entities import ChatMessage, ConversationRef, Evidence
from onyx_desktop.domain.enums import ChatMode, MessageRole, TurnOutcome
from onyx_desktop.ui.widgets.answer_language import AnswerLanguageLabel
from onyx_desktop.ui.widgets.history_list import HistoryEntry, HistoryList
from onyx_desktop.ui.widgets.message_bubble import MessageRow
from onyx_desktop.ui.widgets.source_panel import SourcePanel
from onyx_desktop.utils.formatting import when

_MAX_CLARIFICATION_ROUNDS = 3


class ChatView(QWidget):
    """Owns the transcript, the composer and the source-review panel."""

    chatRequested = Signal(str, object, object, int)   # prompt, session_id, ChatMode, limit
    newConversationRequested = Signal()
    conversationChosen = Signal(object)                 # session_id
    conversationDeleteRequested = Signal(object)        # session_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._session_id = uuid.uuid4()
        self._requested_session: Optional[uuid.UUID] = None
        self._pending_row: Optional[QWidget] = None

        # Reasoned Answer clarification state: the question being refined, and what has been added.
        self._clarifying_question: Optional[str] = None
        self._clarifications: List[str] = []

        self._build()

    # ---------- construction ----------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- far left: past conversations ---
        self._history = HistoryList(
            "Conversations", "No conversations yet. Ask a question to start one.",
            deletable=True, noun="conversation",
        )
        self._history.itemChosen.connect(self._on_history_chosen)
        self._history.deleteRequested.connect(self.conversationDeleteRequested)
        splitter.addWidget(self._history)

        # --- left: conversation ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(24, 20, 16, 20)
        left_layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Chat")
        title.setProperty("role", "heading")
        header.addWidget(title)
        self._answer_language = AnswerLanguageLabel()
        header.addWidget(self._answer_language)
        header.addStretch(1)

        self._new_button = QPushButton("New conversation")
        self._new_button.clicked.connect(self._on_new_conversation)
        header.addWidget(self._new_button)
        left_layout.addLayout(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._transcript = QWidget()
        self._transcript_layout = QVBoxLayout(self._transcript)
        self._transcript_layout.setContentsMargins(0, 0, 8, 0)
        self._transcript_layout.setSpacing(12)
        self._transcript_layout.addStretch(1)
        self._scroll.setWidget(self._transcript)
        left_layout.addWidget(self._scroll, stretch=1)

        left_layout.addWidget(self._composer())

        # --- right: review ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 20, 24, 20)
        right_layout.setSpacing(10)

        review_title = QLabel("Source review")
        review_title.setProperty("role", "heading")
        right_layout.addWidget(review_title)

        hint = QLabel(
            "Every claim should be checkable. Pick a citation to read the words it came from, "
            "and open the recording if you want to hear them."
        )
        hint.setProperty("role", "subtle")
        hint.setWordWrap(True)
        right_layout.addWidget(hint)

        self._source_panel = SourcePanel()
        right_layout.addWidget(self._source_panel, stretch=1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([230, 620, 420])
        outer.addWidget(splitter)

    def _composer(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._input = _Composer()
        self._input.setPlaceholderText(
            "Ask a question about your material…  (Enter to send, Shift+Enter for a new line)"
        )
        self._input.setFixedHeight(84)
        self._input.submitted.connect(self._on_send)
        layout.addWidget(self._input)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self._mode = QComboBox()
        self._mode.addItem("Quick answer", ChatMode.SINGLE_PASS)
        self._mode.addItem("Thorough answer", ChatMode.REASONING)
        self._mode.setToolTip(
            "A thorough answer looks further and double-checks itself. It is much slower and may ask you to "
            "narrow the question first."
        )
        controls.addWidget(self._mode)

        controls.addWidget(QLabel("Sources:"))
        self._limit = QSpinBox()
        # The engine's SearchQuery caps limit at 20; anything higher is a validation error.
        self._limit.setRange(1, 20)
        self._limit.setValue(5)
        self._limit.setToolTip("How many sources the answer may draw on (up to 20)")
        controls.addWidget(self._limit)

        controls.addStretch(1)

        self._clarification_hint = QLabel("")
        self._clarification_hint.setProperty("role", "subtle")
        controls.addWidget(self._clarification_hint)

        self._send = QPushButton("Ask")
        self._send.setProperty("role", "primary")
        self._send.clicked.connect(self._on_send)
        controls.addWidget(self._send)

        layout.addLayout(controls)
        return box

    # ---------- public API ----------

    @property
    def session_id(self) -> uuid.UUID:
        return self._session_id

    def set_answer_language(self, name: str) -> None:
        self._answer_language.set_language(name)

    def set_busy(self, busy: bool) -> None:
        self._send.setEnabled(not busy)
        self._input.setEnabled(not busy)
        self._new_button.setEnabled(not busy)
        self._send.setText("Working…" if busy else "Ask")

    def set_conversations(self, conversations: List[ConversationRef]) -> None:
        self._history.set_entries([
            HistoryEntry(
                key=ref.session_id,
                title=ref.title,
                subtitle=f"{when(ref.last_at)} · {ref.turn_count} turn{'s' if ref.turn_count != 1 else ''}",
            )
            for ref in conversations
        ])
        self._history.select(self._session_id)

    def set_history_unavailable(self, text: str) -> None:
        self._history.set_unavailable(text)

    def show_conversation(self, session_id: uuid.UUID, messages: List[ChatMessage]) -> None:
        """
        Re-renders a past conversation and makes it the current one, so the next question continues it:
        the engine reads the same session's earlier turns and their [Doc N] numbering when answering.
        """
        if session_id != self._requested_session:
            return  # an older request that lost the race to a newer click
        self._session_id = session_id
        self._reset_clarification()
        self._clear_transcript()
        for message in messages:
            self.add_message(message)
            if message.missing_evidence:
                self._add_system_note(
                    f"{message.missing_evidence} source(s) this answer cited are no longer available."
                )
        self._history.select(session_id)

    def forget_conversation(self, session_id: uuid.UUID) -> None:
        """The conversation was deleted; if it is on screen, start a fresh one rather than keep writing to it."""
        if session_id == self._session_id:
            self._on_new_conversation()

    def add_message(self, message: ChatMessage) -> None:
        self._drop_pending()
        row = MessageRow(message)
        row.reviewRequested.connect(self._on_review)
        row.openRequested.connect(self._on_open)
        self._insert(row)

        if message.role is MessageRole.ASSISTANT:
            self._track_clarification(message)
            # Show the first citation straight away: the point is to verify, not to hunt. Even a
            # citation with no file on disk is worth opening -- its verbatim excerpt still is.
            if message.evidence:
                self._source_panel.show_evidence(message.evidence[0])

    def show_failure(self, message: str) -> None:
        self._drop_pending()
        self._add_system_note(message, danger=True)

    # ---------- events ----------

    def _on_send(self) -> None:
        prompt = self._input.toPlainText().strip()
        if not prompt:
            return

        # QComboBox stores userData as a QVariant; a str-derived Enum comes back out as a plain
        # str, so `is` comparisons downstream would silently take the wrong branch. Rebuilt here,
        # at the boundary, so everything below this line gets a real enum.
        mode = ChatMode(self._mode.currentData())
        limit = self._limit.value()

        if self._clarifying_question is not None:
            # A clarification cannot resume the reasoner mid-loop, so the whole question is
            # re-asked with every clarification appended -- the reasoner starts fresh with the
            # fuller picture, matching run_reasoned_with_clarifications in app/main.py.
            self._clarifications.append(prompt)
            outgoing = self._clarifying_question
            for index, clarification in enumerate(self._clarifications, start=1):
                outgoing += f"\n\n[Clarification {index}] {clarification}"
            mode = ChatMode.REASONING
        else:
            outgoing = prompt

        self._input.clear()
        self.add_message(ChatMessage(role=MessageRole.USER, text=prompt))
        self._show_pending(mode)
        self.chatRequested.emit(outgoing, self._session_id, mode, limit)

    def _on_new_conversation(self) -> None:
        self._session_id = uuid.uuid4()
        self._requested_session = None
        self._reset_clarification()
        self._clear_transcript()
        self._history.clear_selection()
        self.newConversationRequested.emit()

    def _on_history_chosen(self, session_id: uuid.UUID) -> None:
        if self._pending_row is not None:
            # An answer for the current conversation is on its way; switching now would drop it into
            # the wrong transcript.
            self._history.select(self._session_id)
            return
        if session_id == self._session_id and self._transcript_layout.count() > 1:
            return
        self._requested_session = session_id
        self.conversationChosen.emit(session_id)

    def _on_review(self, evidence: Evidence) -> None:
        self._source_panel.show_evidence(evidence)

    def _on_open(self, evidence: Evidence) -> None:
        """One click from a card straight to the player; the panel also shows what was opened."""
        self._source_panel.show_evidence(evidence)
        self._source_panel.open_current()

    # ---------- clarification handling ----------

    def _track_clarification(self, message: ChatMessage) -> None:
        if message.outcome is TurnOutcome.NEEDS_CLARIFICATION:
            if self._clarifying_question is None:
                # The question being refined is the last thing the user actually typed.
                self._clarifying_question = self._last_user_text()
            rounds = len(self._clarifications)
            if rounds >= _MAX_CLARIFICATION_ROUNDS:
                self._add_system_note(
                    f"Reached the clarification limit ({_MAX_CLARIFICATION_ROUNDS} rounds) "
                    "without a grounded answer. Try asking a narrower question outright."
                )
                self._reset_clarification()
            else:
                self._clarification_hint.setText(
                    f"clarifying · round {rounds + 1} of {_MAX_CLARIFICATION_ROUNDS}"
                )
        else:
            self._reset_clarification()

    def _reset_clarification(self) -> None:
        self._clarifying_question = None
        self._clarifications = []
        self._clarification_hint.setText("")

    def _last_user_text(self) -> str:
        for index in reversed(range(self._transcript_layout.count())):
            widget = self._transcript_layout.itemAt(index).widget()
            if isinstance(widget, MessageRow):
                bubble = widget.findChild(QWidget)
                text = getattr(widget, "_user_text", None)
                if text:
                    return text
        return ""

    # ---------- transcript plumbing ----------

    def _insert(self, widget: QWidget) -> None:
        # Keep the trailing stretch last so bubbles stack from the top.
        self._transcript_layout.insertWidget(self._transcript_layout.count() - 1, widget)
        self._scroll_to_bottom()

    def _show_pending(self, mode: ChatMode) -> None:
        text = (
            "Working on a thorough answer… this takes a while."
            if mode is ChatMode.REASONING
            else "Answering…"
        )
        label = QLabel(text)
        label.setProperty("role", "subtle")
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._pending_row = label
        self._insert(label)

    def _drop_pending(self) -> None:
        if self._pending_row is None:
            return
        self._pending_row.setParent(None)
        self._pending_row.deleteLater()
        self._pending_row = None

    def _add_system_note(self, text: str, danger: bool = False) -> None:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setProperty("role", "banner-danger" if danger else "banner-warning")
        self._insert(label)

    def _clear_transcript(self) -> None:
        self._drop_pending()
        while self._transcript_layout.count() > 1:
            item = self._transcript_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._source_panel.clear()

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())


class _Composer(QTextEdit):
    """Enter sends, Shift+Enter makes a new line."""

    submitted = Signal()

    def keyPressEvent(self, event) -> None:
        enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        if enter and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submitted.emit()
            return
        super().keyPressEvent(event)
