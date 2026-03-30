# Canvas: frame_extraction_error_mapper_v2
# Path: app/features/frame_extraction/utils/error_mapper.py

import subprocess
from PIL import UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.frame_extraction import exceptions as exc


def map_extraction_error(e: Exception, context: str = "") -> exc.FrameExtractionError:
    """
    Translates raw system, FFmpeg, PIL, and DB exceptions into
    strictly typed Domain Exceptions for the Frame Extraction feature.
    """
    error_msg = str(e).lower()

    if isinstance(e, exc.FrameExtractionError):
        return e

    # 1. FFmpeg Subprocess Errors
    if isinstance(e, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        error_details = getattr(e, "stderr", str(e))
        return exc.FFmpegProcessError(
            f"FFmpeg process failed or timed out: {error_details} | Context: {context}",
            original_exception=e
        )

    # FIX: Added `"video" in error_msg` so it properly catches the adapter's FileNotFoundError
    if isinstance(e, FileNotFoundError) and (
            "ffmpeg" in error_msg or "video" in error_msg or "video" in context.lower()):
        return exc.FFmpegProcessError(
            f"Target video or FFmpeg binary not found: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 2. PIL / I-O Image Processing Errors (Quality & Deduplication)
    if isinstance(e, UnidentifiedImageError) or (isinstance(e, OSError) and "image" in context.lower()):
        if "quality" in context.lower() or "blur" in context.lower():
            return exc.FrameQualityError(
                f"Failed to calculate sharpness/quality for image: {str(e)} | Context: {context}",
                original_exception=e
            )
        return exc.FrameDeduplicationError(
            f"Failed to open, read, or hash the extracted JPEG frame: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        if "chunk" in context.lower() or "frame" in context.lower():
            return exc.ExtractionPersistenceError(
                f"Failed to save ExtractedChunkFrames mapping: {str(e)}",
                original_exception=e
            )
        return exc.ExtractionDatabaseError(
            f"Database operation failed in Frame Extraction: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. Fallback Catch-all
    return exc.FrameExtractionError(
        f"Unexpected failure in Frame Extraction: {str(e)} | Context: {context}",
        original_exception=e
    )