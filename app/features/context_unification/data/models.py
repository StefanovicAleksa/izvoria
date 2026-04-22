# Canvas: context_unification_models_v1
# Path: app/features/context_unification/data/models.py

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base


class UnifiedTranscriptModel(Base):
    """
    SQLAlchemy model for storing the final Phase 15 output.
    This is the ultimate source of truth before data is sent to the Vector Database.
    """
    __tablename__ = "unified_transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    # JSONB List of UnifiedChunk objects
    # Stores: chunk_id, start_sec, end_sec, text_payload, and visual_evidence
    chunks = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )