import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base


class DocumentGlossaryModel(Base):
    """
    SQLAlchemy model for storing the overarching domain terminology (Phase 1 output).
    The 'items' column stores the list of GlossaryItems as optimized JSONB.
    """
    __tablename__ = "document_glossaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    domain_category = Column(String, nullable=False)

    # JSONB List of [{"term": "MRI", "context": "..."}]
    items = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )


class DomainCorrectedTranscriptModel(Base):
    """
    SQLAlchemy model for storing the final Phase 3 output.
    Maintains the corrected grammatical sentences and the strict Before/After Audit Trail.
    """
    __tablename__ = "domain_corrected_transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    # JSONB List of ReconstructedSentence objects (the corrected baseline)
    sentences = Column(JSONB, nullable=False)

    # JSONB List of CorrectionAuditRecord objects (the proof of no hallucination)
    audit_trail = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )