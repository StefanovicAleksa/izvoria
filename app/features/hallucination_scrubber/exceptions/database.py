# Canvas: scrubber_exceptions_database_v1
# Path: app/features/hallucination_scrubber/exceptions/database.py

from .base import HallucinationScrubberError

class ScrubbingDatabaseError(HallucinationScrubberError):
    """Generic database failure during the hallucination scrubbing pipeline."""
    pass

class ScrubbedTranscriptSaveError(ScrubbingDatabaseError):
    """Raised when the final scrubbed transcript or its immutable audit trail fails to persist to PostgreSQL."""
    pass