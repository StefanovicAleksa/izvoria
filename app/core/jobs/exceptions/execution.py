# File: app/core/jobs/exceptions/execution.py

from .base import JobError

class JobDataError(JobError):
    """
    Raised when the data provided to the job system is invalid
    (e.g., trying to attach a job to a non-existent Source ID).
    """
    pass

class JobExecutionFailed(JobError):
    """
    Raised when the actual feature function (like Transcribe or Ingest)
    crashes while being run by the job manager.
    """
    pass