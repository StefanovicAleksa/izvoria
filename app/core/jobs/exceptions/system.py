# File: app/core/jobs/exceptions/system.py

from .base import JobError

class JobSystemError(JobError):
    """
    Raised when the job machinery itself fails (e.g., Database goes offline,
    cannot acquire lock, failure to update job status).
    """
    pass