from .base import MediaCompactionError


class FFmpegNotFoundError(MediaCompactionError):
    """Raised when the ffmpeg or ffprobe binary is not found on the host system."""
    pass


class FFmpegExecutionError(MediaCompactionError):
    """Raised when ffmpeg or ffprobe exits non-zero (e.g. a corrupted recording)."""
    pass


class FFmpegTimeoutError(MediaCompactionError):
    """Raised when a transcode exceeds the maximum allowed time."""
    pass


class ProbeParseError(MediaCompactionError):
    """Raised when ffprobe's output cannot be read."""
    pass
