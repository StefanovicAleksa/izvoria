# Path: app/features/session_memory/service/session_memory_api.py

import logging
import re
import uuid
from typing import Optional

from app.features.session_memory.domain.interfaces import ISessionMemoryRepository
from app.features.session_memory.domain.entities import SessionTurn
from app.features.session_memory.utils.error_mapper import map_session_memory_error

logger = logging.getLogger(__name__)

# [Doc N] tags number the evidence of the turn that produced them; in the next turn they point at other chunks
_CITATION_TAG = re.compile(r"\s*\[\s*Docs?\b[^\[\]]*\]", re.IGNORECASE)


def _strip_citation_tags(text: str) -> str:
    return _CITATION_TAG.sub("", text).strip()


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
        # Turns stored before tags were stripped on write still carry them
        return f"[PRIOR TURN]\nQ: {last_turn.question}\nA: {_strip_citation_tags(last_turn.answer_text)}"

    def record_turn(self, session_id: uuid.UUID, question: str, answer_text: str) -> SessionTurn:
        try:
            return self._repo.record_turn(session_id, question, _strip_citation_tags(answer_text))
        except Exception as e:
            raise map_session_memory_error(e)
