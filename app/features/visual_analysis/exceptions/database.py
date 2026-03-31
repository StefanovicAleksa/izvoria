# Canvas: visual_analysis_exceptions_database_v1
# Path: app/features/visual_analysis/exceptions/database.py

from .base import VisualAnalysisError

class VisualDatabaseError(VisualAnalysisError):
    """Generic database failure during the visual analysis pipeline."""
    pass

class VisualPersistenceError(VisualDatabaseError):
    """Raised when the ChunkVisualAnalysis fails to serialize to JSONB or persist to PostgreSQL."""
    pass