import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import AliasChoices, BaseModel, Field, ConfigDict, computed_field, field_validator

from app.core.jobs.domain.enums import JobStatus, JobType

class JobRequest(BaseModel):
    job_type: JobType

    # only null for the source scanning job
    source_id: Optional[uuid.UUID] = Field(default=None, description="The UUID of the Source being processed")

    input_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

class JobResult(BaseModel):
    status: JobStatus
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None


class Job(BaseModel):
    id: uuid.UUID
    job_type: JobType

    # Foreign Key to the Storage Feature (Source)
    source_id: Optional[uuid.UUID] = None

    status: JobStatus
    worker_id: Optional[str] = None

    # Payloads
    input_params: Dict[str, Any] = Field(default_factory=dict)
    # Persisted in the `output_params` column of the jobs table
    output_result: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("output_params", "output_result")
    )

    # Error Details
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("output_result", mode="before")
    @classmethod
    def _default_missing_output(cls, value: Any) -> Any:
        # A freshly created row has no output yet (the column only has a server default)
        return {} if value is None else value

    @computed_field
    @property
    def duration_ms(self) -> Optional[int]:
        """Calculates duration on-the-fly for API responses."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None