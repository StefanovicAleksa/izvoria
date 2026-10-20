# Path: app/features/source_profile/data/models.py

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.data.engine import Base


class SourceProfileModel(Base):
    """
    What a source is about, one row per source: a title people would refer to it by and a one-sentence topic.
    Deleting the source (re-ingesting or removing the file) deletes its profile.
    """
    __tablename__ = "source_profiles"

    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), primary_key=True
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    # "generated" (a model read the source), "named" (taken from the file name) or "edited" (a person wrote it)
    origin: Mapped[str] = mapped_column(String(16), nullable=False)
    # The language the title and topic are written in: the source's own
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )
