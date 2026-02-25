# Canvas: diarization_exceptions_database_v1
# Path: app/features/diarization/exceptions/database.py

from app.features.diarization.exceptions.base import DiarizationError

class DiarizationDatabaseError(DiarizationError):
    """Raised when the Postgres database fails to save or retrieve the DiarizationMap."""
    pass