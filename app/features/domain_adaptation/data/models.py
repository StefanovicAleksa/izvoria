# Canvas: domain_adaptation_models_v1
# Path: app/features/domain_adaptation/data/models.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database.data.engine import Base


class DomainGlossaryModel(Base):
    """
    SQLAlchemy model for storing the output of the CPU Domain Adaptation phase.
    Stores the exact weighted hotwords injected into the heavy GPU transcriber.
    """
    __tablename__ = "domain_glossaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links directly back to the original media file in the Storage feature
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    industry = Column(String, nullable=False)
    sub_domain = Column(String, nullable=False)

    # Text column to allow for potentially massive hotword strings
    hotword_string = Column(Text, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )