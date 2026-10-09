import json
import subprocess

from app.features.media_compaction import exceptions as compaction_exc


def map_ffmpeg_error(e: Exception) -> compaction_exc.MediaCompactionError:
    """
    Translates raw system and subprocess exceptions into the Media Compaction feature's typed exceptions.
    """
    if isinstance(e, compaction_exc.MediaCompactionError):
        return e

    # 1. ffmpeg / ffprobe binary not found
    if isinstance(e, FileNotFoundError):
        return compaction_exc.FFmpegNotFoundError(
            "ffmpeg/ffprobe not found on the system path. Please ensure they are installed.",
            original_exception=e
        )

    # 2. Transcode timeout
    if isinstance(e, subprocess.TimeoutExpired):
        return compaction_exc.FFmpegTimeoutError(
            f"Media compaction timed out after {e.timeout} seconds.",
            original_exception=e
        )

    # 3. ffmpeg crash / corrupted file / bad arguments
    if isinstance(e, subprocess.CalledProcessError):
        error_details = e.stderr if e.stderr else str(e)
        return compaction_exc.FFmpegExecutionError(
            f"FFmpeg process failed with exit code {e.returncode}. Details: {error_details}",
            original_exception=e
        )

    # 4. ffprobe printed something that isn't its JSON
    if isinstance(e, (json.JSONDecodeError, ValueError, TypeError)):
        return compaction_exc.ProbeParseError(
            f"Could not read ffprobe's output: {str(e)}",
            original_exception=e
        )

    # 5. OS level I/O errors (disk full, permission denied)
    if isinstance(e, OSError):
        return compaction_exc.MediaCompactionError(
            f"System I/O error during media compaction: {str(e)}",
            original_exception=e
        )

    # 6. Fallback
    return compaction_exc.MediaCompactionError(
        f"An unexpected error occurred during media compaction: {str(e)}",
        original_exception=e
    )
