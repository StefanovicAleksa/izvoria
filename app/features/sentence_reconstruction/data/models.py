# Canvas: sentence_models_v2
# Path: app/features/sentence_reconstruction/data/models.py

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base


class ReconstructedTranscriptModel(Base):
    """
    PostgreSQL persistence for Phase 7 outputs.
    Utilizes JSONB for high-performance retrieval of nested RAG documents.
    """
    __tablename__ = "reconstructed_transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # FIXED: Added explicit ForeignKey and unique=True to match other features
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    # FIXED: Renamed from 'data' to 'sentences' for parity with TranscriptModel
    sentences = Column(JSONB, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)