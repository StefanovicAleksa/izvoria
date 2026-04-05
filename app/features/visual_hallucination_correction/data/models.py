# Canvas: visual_hallucination_models_v1
# Path: app/features/visual_hallucination_correction/data/models.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base

class VisualHallucinationEvaluationModel(Base):
    """
    SQLAlchemy model for storing the ledger of visual evaluations.
    Maintains a 1-to-1 mapping with a specific Video Source.
    """
    __tablename__ = "visual_hallucination_evaluations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )

    # JSONB List of serialized VisualHallucinationAudit objects
    audits = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )