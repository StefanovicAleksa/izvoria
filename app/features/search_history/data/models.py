# Path: app/features/search_history/data/models.py

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.data.engine import Base


class SearchHistoryModel(Base):
    """
    Past standard and verified searches. chunk_ids is intentionally not a foreign key (as with
    research_report_evidence.chunk_id): re-ingesting a file cascades its chunks away, and the history of what was
    searched must not go with them.
    """
    __tablename__ = "search_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(Text, nullable=False)
    result_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_ids: Mapped[List[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default=text("'{}'::uuid[]")
    )
    scores: Mapped[List[float]] = mapped_column(
        ARRAY(DOUBLE_PRECISION), nullable=False, server_default=text("'{}'::double precision[]")
    )
    execution_time_ms: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    # The user who searched, on the hosted multi-user API; NULL on the appliance, where history is shared.
    owner_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    __table_args__ = (
        Index("idx_search_history_created_at", "created_at"),
    )
