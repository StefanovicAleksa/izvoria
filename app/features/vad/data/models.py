# File: app/features/vad/data/models.py
from datetime import datetime, timezone
import uuid

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

# Assuming Base is located here per your core database setup
from app.core.database.data.engine import Base


class VadMapModel(Base):
    """
    SQLAlchemy model for storing the VAD Virtual Map.
    Stores the complex list of speech segments as highly optimized JSONB.
    """
    __tablename__ = "vad_maps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, unique=True)

    # The actual virtual map: [{"start_sec": 1.5, "end_sec": 4.2, "confidence": 0.98}, ...]
    segments = Column(JSONB, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Note: Depending on your exact core setup, you may want to add a back-populates
    # relationship to the SourceModel here later, but this is all VAD strictly needs.