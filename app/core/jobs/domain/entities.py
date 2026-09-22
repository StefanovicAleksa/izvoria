import threading
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import AliasChoices, BaseModel, Field, ConfigDict, computed_field, field_validator

from app.core.jobs.domain.enums import JobStatus, JobType

class CancellationToken:
    """
    A cooperative stop signal for long-running work.

    Deliberately not a pydantic model: it is shared mutable state, set from one thread (a GUI, a CLI
    handler) and polled from another (the pipeline). `threading.Event` is the whole implementation --
    `set()` is safe to call from any thread, which is what lets a Stop button work without the caller
    having to reach into the worker thread that is busy.

    Cancellation is checked at unit boundaries, never mid-generation: llama.cpp's completion call is
    blocking with no callback, so the finest granularity available is "between sections".
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self, context: str = "") -> None:
        if self._event.is_set():
            from app.core.jobs.exceptions.execution import JobCancelled
            raise JobCancelled(f"Work was cancelled by the user.{f' | Context: {context}' if context else ''}")


class JobRequest(BaseModel):
    job_type: JobType

    # only null for the source scanning job
    source_id: Optional[uuid.UUID] = Field(default=None, description="The UUID of the Source being processed")

    input_params: Dict[str, Any] = Field(default_factory=dict)

    # A human label for the progress listener, e.g. "Writing section 4/8: Regulatory exposure". Without it
    # a listener only ever sees the job type, which is enough for a spinner but not for a progress bar.
    progress_detail: Optional[str] = Field(default=None)

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