# Canvas: visual_synthesis_models_v1
# Path: app/features/visual_context_synthesis/data/models.py

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base

class VisualTimelineModel(Base):
    """
    SQLAlchemy model for storing the finalized Phase 14 output.
    This acts as the source of truth for all visual-context metadata.
    """
    __tablename__ = "rag_visual_timelines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Direct relationship to the source media
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    # JSONB List of SynthesizedVisualChunk objects
    # Stores: chunk_id, visual_context_string, and visual_evidence (frame pointers)
    visual_chunks = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )