# Path: app/features/session_memory/domain/entities.py

import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.features.session_memory.domain.enums import SessionTurnMode


class SessionTurn(BaseModel):
    """One recorded question/answer turn of a conversation, keyed by session_id.

    evidence_chunk_ids are the chunks this turn showed for the first time; they carry the conversation-wide
    numbers [Doc first_doc_number], [Doc first_doc_number + 1], ... Chat answers keep those tags; Reasoned Answer turns are stored
    without tags, since its answers number their evidence on their own."""
    id: uuid.UUID
    session_id: uuid.UUID
    question: str
    answer_text: str
    mode: SessionTurnMode = SessionTurnMode.CHAT
    standalone_query: Optional[str] = None
    evidence_chunk_ids: List[uuid.UUID] = Field(default_factory=list)
    first_doc_number: int = 1
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @property
    def next_doc_number(self) -> int:
        """The number the next turn's first new document gets."""
        return self.first_doc_number + len(self.evidence_chunk_ids)
