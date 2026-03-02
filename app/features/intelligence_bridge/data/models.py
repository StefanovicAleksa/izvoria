import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base

class SceneMapModel(Base):
    """
    SQLAlchemy model for storing the final deduplicated Intelligent Scene Map.
    Reuses the 'scene_maps' table name for seamless transition.
    """
    __tablename__ = "scene_maps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    # The actual virtual map enriched with semantic data:
    # [{"timestamp_sec": 12.5, "extracted_file_id": "...", "phash_value": "...", "extraction_method": "semantic_pass", "semantic_context": "..."}]
    scenes = Column(JSONB, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )