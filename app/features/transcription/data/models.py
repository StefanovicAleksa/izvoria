# filename: transcription_models_v1.py
# path: app/features/transcription/data/models.py
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base


class TranscriptModel(Base):
    """
    SQLAlchemy model for storing the complete Transcript.
    Stores the list of transcribed segments as highly optimized JSONB.
    """
    __tablename__ = "transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, unique=True)

    # The actual transcript segments: [{"start_sec": 1.5, "end_sec": 4.2, "text": "Hello world", "speaker_id": null}, ...]
    segments = Column(JSONB, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)