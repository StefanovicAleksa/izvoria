# Canvas: scene_detector_exceptions_ffmpeg_v1
# Path: app/features/scene_detector/exceptions/ffmpeg.py

from app.features.scene_detector.exceptions.base import SceneDetectorError

class FFmpegNotFoundError(SceneDetectorError):
    """Raised when the FFmpeg binary is not found on the host system path."""
    pass

class FFmpegExecutionError(SceneDetectorError):
    """Raised when FFmpeg crashes or returns a non-zero exit code (e.g., corrupted video)."""
    pass

class FFmpegTimeoutError(SceneDetectorError):
    """Raised when the scene extraction process exceeds the maximum allowed time."""
    pass

class FFmpegInvalidVideoError(SceneDetectorError):
    """Raised when the target video path does not exist or cannot be read by FFmpeg."""
    pass