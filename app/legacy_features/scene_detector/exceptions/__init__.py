# Canvas: scene_detector_exceptions_init_v1
# Path: app/features/scene_detector/exceptions/__init__.py

from .base import SceneDetectorError
from .ffmpeg import (
    FFmpegNotFoundError,
    FFmpegExecutionError,
    FFmpegTimeoutError,
    FFmpegInvalidVideoError
)
from .processing import SceneDetectionWindowError, SceneDeduplicationError

__all__ = [
    "SceneDetectorError",
    "FFmpegNotFoundError",
    "FFmpegExecutionError",
    "FFmpegTimeoutError",
    "FFmpegInvalidVideoError",
    "SceneDetectionWindowError",
    "SceneDeduplicationError",
]