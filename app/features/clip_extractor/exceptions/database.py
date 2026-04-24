# Canvas: clip_extractor_exceptions_database_v1
# Path: app/features/clip_extractor/exceptions/database.py

from .base import ClipExtractorError

class ClipExtractorDatabaseError(ClipExtractorError):
    """
    Raised when the database fails to save or retrieve video clip records.
    """
    pass