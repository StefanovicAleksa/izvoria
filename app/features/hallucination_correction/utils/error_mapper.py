# Canvas: hallucination_correction_error_mapper_v1
# Path: app/features/hallucination_correction/utils/error_mapper.py

from sqlalchemy.exc import SQLAlchemyError
from app.core.database import exceptions as core_db_exc
from app.features.hallucination_correction import exceptions as exc

def map_correction_error(e: Exception) -> exc.HallucinationCorrectionError:
    """
    Translates raw system, validation, and database exceptions into
    typed Domain Exceptions for the Hallucination Correction feature.
    """
    if isinstance(e, exc.HallucinationCorrectionError):
        return e

    error_msg = str(e).lower()

    # 1. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        return exc.CorrectionDatabaseError(
            f"Failed to update Transcript with sanitized segments: {str(e)}",
            original_exception=e
        )

    # 2. Corrupt Timestamps
    if isinstance(e, ValueError) and ("timestamp" in error_msg or "overlap" in error_msg):
        return exc.CorruptSegmentDataError(
            f"Failed to calculate VAD intersections due to corrupt segment timestamps: {str(e)}",
            original_exception=e
        )

    # 3. Catch-all
    return exc.HallucinationCorrectionError(
        f"An unexpected error occurred during Hallucination Correction: {str(e)}",
        original_exception=e
    )