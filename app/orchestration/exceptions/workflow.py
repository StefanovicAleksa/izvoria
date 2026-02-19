# File: app/orchestration/exceptions/workflow.py

from .base import OrchestrationError

class WorkflowExecutionError(OrchestrationError):
    """
    Raised when a critical, unrecoverable failure occurs during a workflow
    (e.g., inability to connect to the JobManager or Database).
    """
    pass

class ScannerInitializationError(OrchestrationError):
    """
    Raised when the initial directory scan cannot be started
    (e.g., the user provided a completely invalid path).
    """
    pass