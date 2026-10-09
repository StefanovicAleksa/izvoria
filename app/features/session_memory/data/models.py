# Path: app/features/session_memory/data/models.py

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import DateTime, Integer, String, Text, Index, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.data.engine import Base


class SessionTurnModel(Base):
    """
    A conversation's question/answer log, not tied to `sources` — a conversation is not scoped to a single
    ingested source. evidence_chunk_ids lists the chunks a turn showed for the first time, numbered
    first_doc_number, first_doc_number + 1, ... in the conversation-wide [Doc N] sequence, so later turns can show
    them again under the same tags (even once older turns fall outside the history window).
    """
    __tablename__ = "session_turns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)

    mode: Mapped[str] = mapped_column(Text, nullable=False, server_default="chat")
    standalone_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_chunk_ids: Mapped[List[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default=text("'{}'::uuid[]")
    )

    first_doc_number: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    # The user who asked, on the hosted multi-user API; NULL on the appliance, where history is shared.
    owner_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('idx_session_turns_session_history', 'session_id', 'created_at'),
    )
