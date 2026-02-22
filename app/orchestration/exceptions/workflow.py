# filename: orchestration_exceptions_workflow_v1.py
from app.orchestration.exceptions.base import OrchestrationError

class WorkflowError(OrchestrationError):
    """Base exception for errors in the overall pipeline routing and state management."""
    pass

class WorkflowStateError(WorkflowError):
    """
    Raised when the pipeline attempts an invalid phase transition.
    Example: Trying to run Phase 3 (Audio Extraction) before Phase 2 (Ingest) is complete.
    """
    pass

class WorkflowExecutionError(WorkflowError):
    """
    Raised when the Director itself fails (e.g., inability to connect
    to the underlying JobManager to lock the pipeline).
    """
    pass