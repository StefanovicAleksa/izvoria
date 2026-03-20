# Canvas: scene_detector_error_mapper_v2
# Path: app/features/scene_detector/utils/error_mapper.py

import subprocess

from app.legacy_features.scene_detector import exceptions as exc


def map_scene_detector_error(e: Exception, context: str = "") -> exc.SceneDetectorError:
    """
    Translates raw system and subprocess exceptions into
    highly specific typed Domain Exceptions for the Scene Detector feature.

    Args:
        e: The original exception caught.
        context: Optional string providing more details (e.g., the video path or command).
    """
    error_msg = str(e).lower()

    # 1. FFmpeg Binary Missing
    if isinstance(e, FileNotFoundError) and "ffmpeg" in error_msg:
        return exc.FFmpegNotFoundError(
            "FFmpeg binary not found on the system path. Please ensure it is installed.",
            original_exception=e
        )

    # 2. Target Video Missing
    if isinstance(e, FileNotFoundError):
        return exc.FFmpegInvalidVideoError(
            f"The target video file could not be found or read: {context}",
            original_exception=e
        )

    # 3. FFmpeg Timeout
    if isinstance(e, subprocess.TimeoutExpired):
        return exc.FFmpegTimeoutError(
            f"FFmpeg scene extraction timed out after {e.timeout} seconds for {context}.",
            original_exception=e
        )

    # 4. FFmpeg Crashes (Exit Code > 0)
    if isinstance(e, subprocess.CalledProcessError):
        # Subprocess errors hide the actual FFmpeg crash log in stderr
        error_details = e.stderr if e.stderr else str(e)
        return exc.FFmpegExecutionError(
            f"FFmpeg process failed with exit code {e.returncode}. Details: {error_details}",
            original_exception=e
        )

    # 5. Time Window / Math Errors (Domain Logic)
    if isinstance(e, ValueError) and ("window" in error_msg or "time" in error_msg or "bound" in error_msg):
        return exc.SceneDetectionWindowError(
            f"Invalid time window provided for scene detection: {str(e)}",
            original_exception=e
        )

    # 6. OS I/O Errors (Disk full, Permissions, Deduplication file issues)
    if isinstance(e, OSError):
        return exc.SceneDeduplicationError(
            f"System I/O error during scene extraction or hashing: {str(e)}",
            original_exception=e
        )

    # 7. Fallback Catch-all
    return exc.SceneDetectorError(
        f"An unexpected error occurred in the Scene Detector: {str(e)}",
        original_exception=e
    )