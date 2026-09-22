import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, DateTime, Text, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.data.engine import Base
from app.core.jobs.domain.enums import JobStatus

class JobModel(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    job_type: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # FK to sources table (nullable for system jobs)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id"),
        nullable=True,
        index=True
    )

    status: Mapped[str] = mapped_column(String, default=JobStatus.PENDING.value, index=True)
    worker_id: Mapped[str] = mapped_column(String, nullable=True)

    input_params: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    output_params: Mapped[dict] = mapped_column(JSONB, server_default="{}")

    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    stack_trace: Mapped[str] = mapped_column(Text, nullable=True)


    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Recovery Index: Helps find the history of a specific file quickly
    __table_args__ = (
        Index('idx_jobs_source_history', 'source_id', 'created_at'),
    )

# There are no migrations, so renamed enum values are backfilled here and run by OnyxEngine.start() after
# create_all(). Job.job_type is parsed into JobType, so a row left on a retired value makes get_job_history
# raise for that source. Every statement must be idempotent -- this runs on every boot.
JOBS_UPGRADE_DDL = (
    # JobType.DEEP_RESEARCH_REASONING became JobType.REASONING_TURN when the mode was renamed.
    "UPDATE jobs SET job_type = 'reasoning_turn' WHERE job_type = 'deep_research_reasoning'",
)
