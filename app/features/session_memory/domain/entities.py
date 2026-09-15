# Path: app/features/session_memory/domain/entities.py

import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class SessionTurn(BaseModel):
    """One recorded question/answer turn, keyed by session_id. Only the most recent turn per
    session is ever read back (see ISessionMemoryRepository.get_last_turn) — this is
    cross-question memory, distinct from the intra-question evidence_pool/decision_history a
    single chat_deep_research() call builds up and discards."""
    id: uuid.UUID
    session_id: uuid.UUID
    question: str
    answer_text: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
