# Path: app/features/session_memory/service/session_memory_api.py

import logging
import uuid
from typing import Optional

from app.features.session_memory.domain.interfaces import ISessionMemoryRepository
from app.features.session_memory.domain.entities import SessionTurn
from app.features.session_memory.utils.error_mapper import map_session_memory_error

logger = logging.getLogger(__name__)


class SessionMemoryAPI:
    """
    Cross-question memory: what the user asked last turn and what was answered, fed into the
    Reasoner's first iteration of the *next* question. Distinct from the intra-question
    evidence_pool/decision_history a single chat_deep_research() call builds and discards.
    """

    def __init__(self, repository: ISessionMemoryRepository):
        self._repo = repository

    def get_prior_context(self, session_id: uuid.UUID) -> Optional[str]:
        try:
            last_turn = self._repo.get_last_turn(session_id)
        except Exception as e:
            raise map_session_memory_error(e)

        if not last_turn:
            return None
        return f"[PRIOR TURN]\nQ: {last_turn.question}\nA: {last_turn.answer_text}"

    def record_turn(self, session_id: uuid.UUID, question: str, answer_text: str) -> SessionTurn:
        try:
            return self._repo.record_turn(session_id, question, answer_text)
        except Exception as e:
            raise map_session_memory_error(e)
