# Path: app/features/session_memory/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.features.session_memory.domain.entities import SessionSummary, SessionTurn
from app.features.session_memory.domain.enums import SessionTurnMode


class ISessionMemoryRepository(ABC):
    @abstractmethod
    def record_turn(
        self,
        session_id: uuid.UUID,
        question: str,
        answer_text: str,
        mode: SessionTurnMode = SessionTurnMode.CHAT,
        standalone_query: Optional[str] = None,
        evidence_chunk_ids: Optional[List[uuid.UUID]] = None,
        first_doc_number: int = 1,
        owner_id: Optional[str] = None
    ) -> SessionTurn:
        """Persists one question/answer turn for the session."""
        pass

    @abstractmethod
    def get_last_turn(self, session_id: uuid.UUID) -> Optional[SessionTurn]:
        """Returns the most recently recorded turn for the session, or None if it has none."""
        pass

    @abstractmethod
    def get_history(self, session_id: uuid.UUID, limit: int, owner_id: Optional[str] = None) -> List[SessionTurn]:
        """Returns the session's most recent `limit` turns, oldest first (only `owner_id`'s, when given)."""
        pass

    @abstractmethod
    def get_all_turns(self, session_id: uuid.UUID, owner_id: Optional[str] = None) -> List[SessionTurn]:
        """Returns every turn of the session, oldest first."""
        pass

    @abstractmethod
    def list_sessions(self, limit: int, owner_id: Optional[str] = None) -> List[SessionSummary]:
        """Returns the `limit` most recently active conversations, most recent first."""
        pass

    @abstractmethod
    def delete_session(self, session_id: uuid.UUID, owner_id: Optional[str] = None) -> int:
        """Removes every turn of the session; returns how many there were."""
        pass
