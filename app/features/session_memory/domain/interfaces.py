# Path: app/features/session_memory/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.features.session_memory.domain.entities import SessionTurn


class ISessionMemoryRepository(ABC):
    @abstractmethod
    def record_turn(self, session_id: uuid.UUID, question: str, answer_text: str) -> SessionTurn:
        """Persists one question/answer turn for the session."""
        pass

    @abstractmethod
    def get_last_turn(self, session_id: uuid.UUID) -> Optional[SessionTurn]:
        """Returns the most recently recorded turn for the session, or None if it has none."""
        pass
