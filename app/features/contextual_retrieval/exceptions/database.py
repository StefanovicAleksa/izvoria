# Canvas: contextual_exceptions_database_v1
# Path: app/features/contextual_retrieval/exceptions/database.py

from .base import ContextualRetrievalError

class ContextualDatabaseError(ContextualRetrievalError):
    """Generic database failure during the contextual retrieval pipeline."""
    pass

class ContextualPersistenceError(ContextualDatabaseError):
    """Raised when the ContextualizedTranscript fails to serialize to JSONB or persist to PostgreSQL."""
    pass