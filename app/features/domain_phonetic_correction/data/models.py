# Canvas: domain_phonetic_models_v2
# Path: app/features/domain_phonetic_correction/data/models.py

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base

class DomainPhoneticCorrectedTranscriptModel(Base):
    __tablename__ = "domain_phonetic_corrected_transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    sentences = Column(JSONB, nullable=False)
    audit_trail = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )