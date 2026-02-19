# File: app/orchestration/exceptions/base.py

class OrchestrationError(Exception):
    """
    The Root Exception for the Orchestration Layer.
    All other orchestration exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception