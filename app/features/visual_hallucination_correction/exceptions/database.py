# Canvas: visual_hallucination_exceptions_database_v1
# Path: app/features/visual_hallucination_correction/exceptions/database.py

from .base import VisualHallucinationError

class VisualHallucinationDatabaseError(VisualHallucinationError):
    """Generic database failure during the hallucination correction pipeline."""
    pass

class AuditPersistenceError(VisualHallucinationDatabaseError):
    """Raised when the final Evaluated Transcript or its audit ledger fails to persist to PostgreSQL."""
    pass