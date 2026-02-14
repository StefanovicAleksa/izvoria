import uuid
from datetime import datetime, timezone
from sqlalchemy import String, ForeignKey, Boolean, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database.data.engine import Base


class SourceModel(Base):
    __tablename__ = "sources"

    # Identity
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Metadata
    name: Mapped[str] = mapped_column(String, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False)

    # DEDUPLICATION KEY: This is the most important column in the system.
    # We index it because every upload checks this first.
    content_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                                                 onupdate=lambda: datetime.now(timezone.utc))

    # Relationship: One-to-One
    # uselist=False ensures we get a single object, not a list.
    # cascade="all, delete-orphan" means if we delete the Source, the File info is also deleted.
    file: Mapped["FileModel"] = relationship(
        back_populates="source",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="joined"  # Performance: Always fetch the file info when fetching a source
    )


class FileModel(Base):
    __tablename__ = "files"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Key linking to Source
    source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sources.id"), unique=True, nullable=False, index=True)

    # Physical File Attributes
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=True)

    # Flexible Metadata (e.g., {"duration": 120, "width": 1920})
    file_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default='{}')

    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Back-reference
    source: Mapped["SourceModel"] = relationship(back_populates="file")