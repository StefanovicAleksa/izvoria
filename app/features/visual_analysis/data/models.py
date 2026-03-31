# Canvas: visual_analysis_models_v1
# Path: app/features/visual_analysis/data/models.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base


class ChunkVisualAnalysisModel(Base):
    """
    SQLAlchemy model for storing the VLM analysis results (Phase 10).
    Maintains a 1-to-1 mapping with a ContextualizedChunk via the chunk_id.
    """
    __tablename__ = "chunk_visual_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Links to the logical ContextualizedChunk.
    # It is indexed for fast lookups and unique to enforce our crash-recovery idempotency rule.
    chunk_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        unique=True,
        index=True
    )

    # JSONB List of serialized FrameAnalysis objects
    analyses = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )