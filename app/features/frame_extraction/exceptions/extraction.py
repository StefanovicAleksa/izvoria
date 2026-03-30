# Canvas: frame_extraction_exceptions_extraction_v1
# Path: app/features/frame_extraction/exceptions/extraction.py

from .base import FrameExtractionError

class FFmpegProcessError(FrameExtractionError):
    """Raised when FFmpeg crashes, times out, or fails to read the video."""
    pass

class FrameQualityError(FrameExtractionError):
    """Raised when the blur/motion detection logic fails to process a frame."""
    pass

class FrameDeduplicationError(FrameExtractionError):
    """Raised when the pHash utility fails due to corrupted JPEGs or I/O issues."""
    pass