# Path: app/features/session_memory/exceptions/database.py

from .base import SessionMemoryError

class SessionMemoryDataError(SessionMemoryError):
    """Raised when invalid data is provided to the session memory system."""
    pass

class SessionMemorySystemError(SessionMemoryError):
    """Raised when the session memory system's underlying database is unavailable."""
    pass
