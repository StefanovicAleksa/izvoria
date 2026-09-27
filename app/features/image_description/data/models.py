# Path: app/features/image_description/data/models.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.core.database.data.engine import Base


class ImageDescriptionModel(Base):
    """What the vision model and OCR found in one image, one row per source."""
    __tablename__ = "image_descriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    file_name = Column(String, nullable=False)
    shows = Column(Text, nullable=False, default="")
    marks = Column(Text, nullable=True)
    ocr_text = Column(Text, nullable=False, default="")
    ocr_lines_detected = Column(Integer, nullable=False, default=0)
    ocr_lines_read = Column(Integer, nullable=False, default=0)
    ocr_checked = Column(Boolean, nullable=False, default=False)
    # JSONB lists of strings
    verified_items = Column(JSONB, nullable=False, default=list)
    unverified_items = Column(JSONB, nullable=False, default=list)
    dropped_items = Column(JSONB, nullable=False, default=list)
    taken_at = Column(String, nullable=True)
    width = Column(Integer, nullable=False, default=0)
    height = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
