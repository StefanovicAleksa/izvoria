# filename: app/features/audio_extractor/exceptions/__init__.py

from .base import AudioExtractorError, AudioExtractionDatabaseError
from .ffmpeg import FFmpegNotFoundError, FFmpegExecutionError, FFmpegTimeoutError

__all__ = [
    "AudioExtractorError",
    "AudioExtractionDatabaseError",
    "FFmpegNotFoundError",
    "FFmpegExecutionError",
    "FFmpegTimeoutError"
]