# File: app/core/jobs/exceptions/base.py

class JobError(Exception):
    """
    The Root Exception for the Jobs Core Feature.
    All other job exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception