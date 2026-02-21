from .base import AudioExtractorError

class FFmpegNotFoundError(AudioExtractorError):
    """Raised when the FFmpeg binary is not found on the host system."""
    pass

class FFmpegExecutionError(AudioExtractorError):
    """Raised when FFmpeg crashes or returns a non-zero exit code (e.g., corrupted video)."""
    pass

class FFmpegTimeoutError(AudioExtractorError):
    """Raised when the extraction process exceeds the maximum allowed time."""
    pass