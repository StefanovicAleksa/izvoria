# Canvas: domain_adaptation_models_v3
# Path: app/features/domain_adaptation/data/models.py

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database.data.engine import Base


class DomainGlossaryModel(Base):
    """
    SQLAlchemy model for storing the output of the LLM Domain Adaptation phase.
    """
    __tablename__ = "domain_glossaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    industry = Column(String, nullable=False)
    sub_domains = Column(JSONB, nullable=False, server_default='[]')

    # Beam search bias
    hotword_string = Column(Text, nullable=False)

    # Transformer context primer
    initial_prompt_string = Column(Text, nullable=False, server_default='')

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )