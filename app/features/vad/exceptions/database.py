# File: app/features/vad/exceptions/database.py
from app.features.vad.exceptions.base import VadError

class VadDatabaseError(VadError):
    """Raised when the Postgres database fails to save or retrieve the VadMap."""
    pass