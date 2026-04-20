# Canvas: visual_synthesis_exceptions_database_v1
# Path: app/features/visual_context_synthesis/exceptions/database.py

from .base import VisualSynthesisError

class VisualSynthesisDatabaseError(VisualSynthesisError):
    """Generic database connectivity or transaction failure during synthesis."""
    pass

class VisualTimelinePersistenceError(VisualSynthesisDatabaseError):
    """Raised when the fully synthesized VisualTimeline fails to serialize to JSONB or persist to PostgreSQL."""
    pass