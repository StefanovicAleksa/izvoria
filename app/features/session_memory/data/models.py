# Path: app/features/session_memory/data/models.py

import uuid
from datetime import datetime, timezone
from sqlalchemy import DateTime, Text, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.data.engine import Base


class SessionTurnModel(Base):
    """
    A standalone question/answer log, not tied to `sources` — deep research sessions are not
    scoped to a single ingested source. Only the most recent row per session_id is ever read.
    """
    __tablename__ = "session_turns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('idx_session_turns_session_history', 'session_id', 'created_at'),
    )
