# filename: transcription_error_mapper_v1.py
# path: app/features/transcription/utils/error_mapper.py
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.transcription import exceptions as exc


def map_transcription_error(e: Exception) -> exc.TranscriptionError:
    """
    Translates raw system, CTranslate2, Faster-Whisper, and DB exceptions into
    highly specific typed Domain Exceptions for the Transcription feature.
    """
    error_msg = str(e).lower()

    # 1. Local Model Missing Error
    if isinstance(e, FileNotFoundError) or "model path" in error_msg:
        return exc.TranscriptionModelFilePathError(
            f"The Faster-Whisper model directory was not found: {str(e)}",
            original_exception=e
        )

    # 2. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        return exc.TranscriptionDatabaseError(
            f"Failed to persist or retrieve Transcript: {str(e)}",
            original_exception=e
        )

    # 3. Audio / Slicing Errors
    if "audio" in error_msg or "bounds" in error_msg or isinstance(e, ValueError) and "timestamp" in error_msg:
        return exc.TranscriptionAudioError(
            f"Failed to slice or process audio according to the VAD Map: {str(e)}",
            original_exception=e
        )

    # 4. CTranslate2 Engine Errors (Hardware/Quantization issues)
    if "ctranslate2" in error_msg or "compute type" in error_msg or "cublas" in error_msg:
        return exc.CTranslateEngineError(
            f"The CTranslate2 inference engine encountered a hardware or runtime fault: {str(e)}",
            original_exception=e
        )

    # 5. General Faster-Whisper Inference Errors
    if "whisper" in error_msg or isinstance(e, RuntimeError):
        return exc.TranscriptionInferenceError(
            f"The Faster-Whisper model failed during transcription: {str(e)}",
            original_exception=e
        )

    # 6. Fallback Catch-all
    return exc.TranscriptionError(
        f"An unexpected error occurred in the Transcription feature: {str(e)}",
        original_exception=e
    )