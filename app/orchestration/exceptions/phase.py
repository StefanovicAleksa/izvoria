# filename: orchestration_exceptions_phase_v1.py
from app.orchestration.exceptions.base import OrchestrationError

class PhaseError(OrchestrationError):
    """Base exception for errors occurring within a specific execution Phase."""
    pass

class PhaseInitializationError(PhaseError):
    """Raised when a phase cannot start (e.g., missing dependencies, empty input batch)."""
    pass

class ScannerInitializationError(PhaseInitializationError):
    """Raised when the initial directory scan cannot be started (e.g., invalid path)."""
    pass

class PhaseExecutionAborted(PhaseError):
    """
    Raised when a phase encounters a critical infrastructure error
    (like Database disconnection) and must halt the entire batch,
    rather than just failing a single item.
    """
    pass