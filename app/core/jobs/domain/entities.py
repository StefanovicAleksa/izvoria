import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, computed_field

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
    output_result: Dict[str, Any] = Field(default_factory=dict)

    # Error Details
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def duration_ms(self) -> Optional[int]:
        """Calculates duration on-the-fly for API responses."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None