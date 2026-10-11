# Canvas: clip_extractor_models_v1
# Path: app/features/clip_extractor/data/models.py

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

from app.core.database.data.engine import Base

class VideoClipModel(Base):
    """
    SQLAlchemy model for storing the physical clip mappings.
    Links a specific RAG chunk to its extracted .mp4 file.
    """
    __tablename__ = "video_clips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link back to the original full-length video source
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Link to the specific RAG chunk from the Unification phase
    chunk_id = Column(UUID(as_uuid=True), nullable=False, unique=True, index=True)

    # Link to the newly generated physical .mp4 file in the Izvoria Vault
    extracted_file_id = Column(
        UUID(as_uuid=True),
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )