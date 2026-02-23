# File: app/features/vad/utils/error_mapper.py
import soundfile as sf
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.vad import exceptions as exc


def map_vad_error(e: Exception) -> exc.VadError:
    """
    Translates raw system, PyTorch, NeMo, and DB exceptions into
    highly specific typed Domain Exceptions for the VAD feature.
    """
    error_msg = str(e).lower()

    # 1. Local File Missing Error
    if isinstance(e, FileNotFoundError) or "not found" in error_msg:
        return exc.VadModelFilePathError(
            f"The VAD model file (.nemo) was not found at the specified path: {str(e)}",
            original_exception=e
        )

    # 2. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        return exc.VadDatabaseError(
            f"Failed to persist or retrieve VAD Map: {str(e)}",
            original_exception=e
        )

    # 3. Audio Formatting & Sample Rate (Soundfile / Torchaudio)
    if "sample rate" in error_msg or "resample" in error_msg or "hz" in error_msg:
        return exc.AudioSampleRateError(
            f"Audio sample rate mismatch. NeMo VAD requires specific frequencies (usually 16kHz): {str(e)}",
            original_exception=e
        )

    if isinstance(e, sf.LibsndfileError) or "format" in error_msg or "wav" in error_msg or "audio" in error_msg:
        return exc.AudioFormatError(
            f"The audio file is corrupted or unreadable by the processing library: {str(e)}",
            original_exception=e
        )

    # 4. PyTorch Device Mismatches (CPU vs GPU)
    if isinstance(e, RuntimeError) and (
            "expected all tensors to be on the same device" in error_msg or "device" in error_msg):
        return exc.VadDeviceError(
            f"PyTorch device mismatch during inference. Ensure audio tensors are moved to the same device as the model: {str(e)}",
            original_exception=e
        )

    # 5. NeMo Configuration / Restore Errors
    if "omegaconf" in error_msg or "configattributeerror" in error_msg or "hydra" in error_msg or "restore" in error_msg:
        return exc.VadModelConfigError(
            f"NeMo configuration graph is invalid or failed to restore from file: {str(e)}",
            original_exception=e
        )

    # 6. General PyTorch/NeMo Inference Errors
    if isinstance(e, RuntimeError) or "tensor" in error_msg or "nemo" in error_msg or "dimension" in error_msg:
        return exc.VadInferenceError(
            f"The NeMo VAD model failed during the forward pass: {str(e)}",
            original_exception=e
        )

    # 7. Fallback Catch-all
    return exc.VadError(
        f"An unexpected error occurred in the VAD feature: {str(e)}",
        original_exception=e
    )