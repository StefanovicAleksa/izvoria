# Canvas: frame_extraction_models_v1
# Path: app/features/frame_extraction/data/models.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base


class ExtractedChunkFramesModel(Base):
    """
    SQLAlchemy model for storing the mapping of a text chunk to its high-quality visual frames.
    Stores the list of ExtractedFrame objects as highly optimized JSONB.
    """
    __tablename__ = "extracted_chunk_frames"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False
    )

    # Links to the logical ContextualizedChunk from Phase 9.
    # It is indexed for fast lookups and unique to enforce 1:1 mapping.
    chunk_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        unique=True,
        index=True
    )

    # JSONB List of ExtractedFrame objects
    frames = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )