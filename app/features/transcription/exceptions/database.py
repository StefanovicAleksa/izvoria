# filename: transcription_exceptions_database_v1.py
# path: app/features/transcription/exceptions/database.py
from app.features.transcription.exceptions.base import TranscriptionError

class TranscriptionDatabaseError(TranscriptionError):
    """Raised when the Postgres database fails to save or retrieve the Transcript."""
    pass