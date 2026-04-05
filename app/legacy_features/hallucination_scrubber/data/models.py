# Canvas: scrubber_models_v1
# Path: app/features/hallucination_scrubber/data/models.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base


class HallucinationScrubbedTranscriptModel(Base):
    """
    SQLAlchemy model for storing the output of the Hallucination Scrubber phase.
    Stores both the cleaned transcript segments and the immutable audit trail.
    """
    __tablename__ = "hallucination_scrubbed_transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    # Stores the cleaned List[TranscriptSegment]
    segments = Column(JSONB, nullable=False)

    # Stores the List[AppliedScrubbingRecord] to prove compliance
    audit_trail = Column(JSONB, nullable=False, server_default='[]')

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )