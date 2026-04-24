# Canvas: clip_extractor_exceptions_init_v1
# Path: app/features/clip_extractor/exceptions/__init__.py

from .base import ClipExtractorError
from .database import ClipExtractorDatabaseError
from .ffmpeg import (
    FFmpegVideoNotFoundError,
    FFmpegVideoExecutionError,
    FFmpegVideoTimeoutError
)

__all__ = [
    "ClipExtractorError",
    "ClipExtractorDatabaseError",
    "FFmpegVideoNotFoundError",
    "FFmpegVideoExecutionError",
    "FFmpegVideoTimeoutError"
]