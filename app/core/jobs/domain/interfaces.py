from abc import ABC, abstractmethod
from typing import Optional, List
import uuid

from app.core.jobs.domain.entities import Job, JobRequest

class IJobRepository(ABC):
    @abstractmethod
    def create_job(self, request: JobRequest) -> Job:
        """Creates a new job (PENDING)."""
        pass

    @abstractmethod
    def start_job(self, job_id: uuid.UUID) -> Optional[Job]:
        """Transitions PENDING -> PROCESSING and sets started_at."""
        pass

    @abstractmethod
    def finish_job(self, job_id: uuid.UUID, output_data: dict) -> Job:
        """Transitions PROCESSING -> COMPLETED and saves output."""
        pass

    @abstractmethod
    def fail_job(self, job_id: uuid.UUID, error_message: str, stack_trace: Optional[str] = None) -> Job:
        """Transitions PROCESSING -> FAILED and saves error details."""
        pass

    @abstractmethod
    def cancel_job(self, job_id: uuid.UUID) -> Job:
        """Transitions PENDING/PROCESSING -> CANCELLED."""
        pass

    @abstractmethod
    def get_job_history(self, source_id: uuid.UUID) -> List[Job]:
        """Retrieves the full execution history for a specific Source."""
        pass