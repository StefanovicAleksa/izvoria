# Canvas: frame_extraction_exceptions_database_v1
# Path: app/features/frame_extraction/exceptions/database.py

from .base import FrameExtractionError

class ExtractionDatabaseError(FrameExtractionError):
    """Generic database failure during the frame extraction pipeline."""
    pass

class ExtractionPersistenceError(ExtractionDatabaseError):
    """Raised when the ExtractedChunkFrames fails to persist to PostgreSQL."""
    pass