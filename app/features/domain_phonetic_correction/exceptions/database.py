# Canvas: exceptions_database_v1
# Path: app/features/phonetic_correction/exceptions/database.py

from .base import PhoneticCorrectionError

class PhoneticDatabaseError(PhoneticCorrectionError):
    """Generic database failure during the phonetic correction pipeline."""
    pass

class TranscriptSaveError(PhoneticDatabaseError):
    """Raised when the final corrected transcript or its audit trail fails to persist to PostgreSQL."""
    pass