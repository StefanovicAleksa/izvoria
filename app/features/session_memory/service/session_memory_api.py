# Path: app/features/session_memory/service/session_memory_api.py

import logging
import re
import uuid
from typing import List, Optional

from app.core.config import settings
from app.features.session_memory.domain.interfaces import ISessionMemoryRepository
from app.features.session_memory.domain.entities import SessionSummary, SessionTurn
from app.features.session_memory.domain.enums import SessionTurnMode
from app.features.session_memory.utils.error_mapper import map_session_memory_error

logger = logging.getLogger(__name__)

_CITATION_TAG = re.compile(r"\s*\[\s*Docs?\b[^\[\]]*\]", re.IGNORECASE)


def strip_citation_tags(text: str) -> str:
    return _CITATION_TAG.sub("", text).strip()


class SessionMemoryAPI:
    """
    Conversation memory shared by chat() and chat_reasoned(): the session's recent turns, the evidence each
    turn showed first, and the conversation-wide [Doc N] numbering of that evidence.
    """

    def __init__(self, repository: ISessionMemoryRepository):
        self._repo = repository

    def get_history(self, session_id: uuid.UUID) -> List[SessionTurn]:
        """The session's most recent turns (at most CHAT_HISTORY_MAX_TURNS), oldest first."""
        try:
            return self._repo.get_history(session_id, settings.CHAT_HISTORY_MAX_TURNS)
        except Exception as e:
            raise map_session_memory_error(e)

    @staticmethod
    def format_prior_context(turns: List[SessionTurn]) -> Optional[str]:
        """
        Earlier questions and answers as plain text for the query rewriter and the Reasoner. Tags are
        stripped: those models number their evidence differently, so a [Doc N] here would point at something else.
        """
        if not turns:
            return None
        parts = [
            f"[PRIOR TURN {i}]\nQ: {turn.question}\nA: {strip_citation_tags(turn.answer_text)}"
            for i, turn in enumerate(turns, start=1)
        ]
        return "\n\n".join(parts)

    def get_all_turns(self, session_id: uuid.UUID) -> List[SessionTurn]:
        """Every turn of the session, oldest first: for showing a conversation again, not for a prompt."""
        try:
            return self._repo.get_all_turns(session_id)
        except Exception as e:
            raise map_session_memory_error(e)

    def list_sessions(self, limit: int = 200) -> List[SessionSummary]:
        """Conversations, most recently active first."""
        try:
            return self._repo.list_sessions(limit)
        except Exception as e:
            raise map_session_memory_error(e)

    def delete_session(self, session_id: uuid.UUID) -> int:
        try:
            return self._repo.delete_session(session_id)
        except Exception as e:
            raise map_session_memory_error(e)

    def get_prior_context(self, session_id: uuid.UUID) -> Optional[str]:
        return self.format_prior_context(self.get_history(session_id))

    def record_turn(
        self,
        session_id: uuid.UUID,
        question: str,
        answer_text: str,
        mode: SessionTurnMode = SessionTurnMode.CHAT,
        standalone_query: Optional[str] = None,
        evidence_chunk_ids: Optional[List[uuid.UUID]] = None,
        first_doc_number: int = 1
    ) -> SessionTurn:
        """
        Chat answers keep their tags, which already use the conversation-wide numbers. Reasoned Answer turns number
        their evidence on their own, so their tags are removed before they can be mistaken for conversation tags.
        """
        if mode == SessionTurnMode.REASONING:
            answer_text = strip_citation_tags(answer_text)
        try:
            return self._repo.record_turn(
                session_id, question, answer_text,
                mode=mode,
                standalone_query=standalone_query,
                evidence_chunk_ids=list(evidence_chunk_ids or []),
                first_doc_number=first_doc_number
            )
        except Exception as e:
            raise map_session_memory_error(e)
