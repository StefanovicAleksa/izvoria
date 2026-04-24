# Canvas: clip_extractor_exceptions_ffmpeg_v1
# Path: app/features/clip_extractor/exceptions/ffmpeg.py

from .base import ClipExtractorError

class FFmpegVideoNotFoundError(ClipExtractorError):
    """Raised when the FFmpeg binary is not found on the host system."""
    pass

class FFmpegVideoExecutionError(ClipExtractorError):
    """Raised when FFmpeg crashes or returns a non-zero exit code (e.g., corrupted video)."""
    pass

class FFmpegVideoTimeoutError(ClipExtractorError):
    """Raised when the extraction process exceeds the maximum allowed time."""
    pass