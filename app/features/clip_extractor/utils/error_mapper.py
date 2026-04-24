# Canvas: clip_extractor_error_mapper_v1
# Path: app/features/clip_extractor/utils/error_mapper.py

import subprocess
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.clip_extractor import exceptions as exc


def map_clip_extractor_error(e: Exception, context: str = "") -> exc.ClipExtractorError:
    """
    Translates raw system, subprocess, and database exceptions into
    typed Domain Exceptions for the Clip Extractor feature.
    """
    # Guard against double-wrapping
    if isinstance(e, exc.ClipExtractorError):
        return e

    error_msg = str(e).lower()

    # 1. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        return exc.ClipExtractorDatabaseError(
            f"Database operation failed in Clip Extractor: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 2. FFmpeg Binary Not Found / Missing Source File
    if isinstance(e, FileNotFoundError):
        if "ffmpeg" in error_msg:
            return exc.FFmpegVideoNotFoundError(
                f"FFmpeg binary not found on the system path. {context}",
                original_exception=e
            )
        return exc.ClipExtractorError(
            f"File not found during extraction: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. FFmpeg Execution Timeout
    if isinstance(e, subprocess.TimeoutExpired):
        return exc.FFmpegVideoTimeoutError(
            f"Video extraction timed out after {e.timeout} seconds. Context: {context}",
            original_exception=e
        )

    # 4. FFmpeg Crash / Bad Arguments
    if isinstance(e, subprocess.CalledProcessError):
        error_details = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
        return exc.FFmpegVideoExecutionError(
            f"FFmpeg process failed with exit code {e.returncode}. Details: {error_details} | Context: {context}",
            original_exception=e
        )

    # 5. OS Level I/O Errors (e.g., Disk full)
    if isinstance(e, OSError):
        return exc.ClipExtractorError(
            f"System I/O error during video extraction: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 6. Fallback Catch-all
    return exc.ClipExtractorError(
        f"An unexpected error occurred in the Clip Extractor: {str(e)} | Context: {context}",
        original_exception=e
    )