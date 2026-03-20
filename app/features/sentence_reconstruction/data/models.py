# Canvas: sentence_models_v3
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
    Because JSONB is schema-less, it seamlessly handles our architectural shift
    from acoustic 'segments' to absolute 'words' without a database migration.
    """
    __tablename__ = "reconstructed_transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    # Stores the List[ReconstructedSentence] as a deeply nested JSON document
    sentences = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )