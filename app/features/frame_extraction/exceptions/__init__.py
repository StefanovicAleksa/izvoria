# Canvas: frame_extraction_exceptions_init_v1
# Path: app/features/frame_extraction/exceptions/__init__.py

from .base import FrameExtractionError
from .extraction import FFmpegProcessError, FrameQualityError, FrameDeduplicationError
from .database import ExtractionDatabaseError, ExtractionPersistenceError

__all__ = [
    "FrameExtractionError",
    "FFmpegProcessError",
    "FrameQualityError",
    "FrameDeduplicationError",
    "ExtractionDatabaseError",
    "ExtractionPersistenceError"
]