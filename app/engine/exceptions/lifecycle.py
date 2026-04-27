# Canvas: engine_exceptions_lifecycle_v1
# Path: app/engine/exceptions/lifecycle.py

from app.engine.exceptions.base import EngineError

class EngineOfflineError(EngineError):
    """Raised when an operation is requested but the engine has not been started via start()."""
    pass

class EngineBusyError(EngineError):
    """
    Raised when requesting an operation that conflicts with current hardware locks.
    (e.g., trying to run a RAG Search while a massive Video Batch Ingestion is hogging the GPU).
    """
    pass

class EngineBootError(EngineError):
    """Raised when the bootstrapper fails to initialize the dependencies (e.g., DB down, models missing)."""
    pass