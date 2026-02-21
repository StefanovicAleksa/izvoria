import subprocess
from app.features.audio_extractor import exceptions as extractor_exc

def map_ffmpeg_error(e: Exception) -> extractor_exc.AudioExtractorError:
    """
    Translates raw system and subprocess exceptions into Domain-Specific FFmpeg Exceptions.
    """
    # 1. FFmpeg Binary Not Found
    if isinstance(e, FileNotFoundError):
        return extractor_exc.FFmpegNotFoundError(
            "FFmpeg binary not found on the system path. Please ensure it is installed.",
            original_exception=e
        )

    # 2. FFmpeg Execution Timeout
    if isinstance(e, subprocess.TimeoutExpired):
        return extractor_exc.FFmpegTimeoutError(
            f"Audio extraction timed out after {e.timeout} seconds.",
            original_exception=e
        )

    # 3. FFmpeg Crash / Corrupted File / Bad Arguments
    if isinstance(e, subprocess.CalledProcessError):
        # Extract the actual FFmpeg crash log from stderr if available
        error_details = e.stderr if e.stderr else str(e)
        return extractor_exc.FFmpegExecutionError(
            f"FFmpeg process failed with exit code {e.returncode}. Details: {error_details}",
            original_exception=e
        )

    # 4. OS Level I/O Errors (e.g., Disk full, permission denied on execution)
    if isinstance(e, OSError):
        return extractor_exc.AudioExtractorError(
            f"System I/O error during FFmpeg execution: {str(e)}",
            original_exception=e
        )

    # 5. Fallback
    return extractor_exc.AudioExtractorError(
        f"An unexpected error occurred during audio extraction: {str(e)}",
        original_exception=e
    )