# Canvas: context_unification_exceptions_database_v1
# Path: app/features/context_unification/exceptions/database.py

from .base import ContextUnificationError

class ContextUnificationDatabaseError(ContextUnificationError):
    """Generic database connectivity or transaction failure during context unification."""
    pass

class UnifiedTranscriptPersistenceError(ContextUnificationDatabaseError):
    """Raised when the fully assembled UnifiedTranscript fails to serialize to JSONB or persist to PostgreSQL."""
    pass