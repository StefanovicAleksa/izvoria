# Canvas: phonetic_models_v1
# Path: app/features/phonetic_correction/data/models.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base


class PhoneticCorrectedTranscriptModel(Base):
    """
    SQLAlchemy model for storing the final Phase 8 output.
    Maintains the corrected grammatical sentences and the strict Before/After Audit Trail.
    """
    __tablename__ = "phonetic_corrected_transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    # JSONB List of ReconstructedSentence objects (the corrected baseline for RAG)
    sentences = Column(JSONB, nullable=False)

    # JSONB List of AppliedCorrectionRecord objects (the proof of no hallucination)
    audit_trail = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )