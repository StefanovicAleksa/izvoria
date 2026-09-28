# Path: app/features/document_conversion/data/models.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base


class ConvertedDocumentModel(Base):
    """A document's page-tagged Markdown, one row per source."""
    __tablename__ = "converted_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    kind = Column(String, nullable=False)
    # JSONB list of PageContent
    pages = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
