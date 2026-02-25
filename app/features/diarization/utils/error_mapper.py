# Canvas: diarization_error_mapper_v1
# Path: app/features/diarization/utils/error_mapper.py

from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.diarization import exceptions as exc


def map_diarization_error(e: Exception) -> exc.DiarizationError:
    """
    Translates raw system, PyTorch, NeMo, SciPy (clustering), and DB exceptions into
    highly specific typed Domain Exceptions for the Diarization feature.
    """
    error_msg = str(e).lower()

    # 1. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        return exc.DiarizationDatabaseError(
            f"Failed to persist or retrieve Diarization Map: {str(e)}",
            original_exception=e
        )

    # 2. Audio Too Short for Embeddings
    if "too short" in error_msg or "minimum duration" in error_msg or "insufficient samples" in error_msg:
        return exc.DiarizationAudioSegmentError(
            f"Audio segments provided by VAD are too short for TitaNet to generate voiceprints: {str(e)}",
            original_exception=e
        )

    # 3. Spectral Clustering / Linear Algebra Failures
    # SciPy and NumPy throw LinAlgError when eigenvalue decomposition fails during Spectral Clustering
    if "linalgerror" in error_msg or "singular matrix" in error_msg or "converge" in error_msg or "clustering" in error_msg:
        return exc.DiarizationClusteringError(
            f"Spectral Clustering failed to group speakers. The audio may be too noisy or speech is overlapping: {str(e)}",
            original_exception=e
        )

    # 4. Zero Speakers Detected Edge Case
    if "0 speakers" in error_msg or "zero clusters" in error_msg:
        return exc.DiarizationZeroSpeakersError(
            "The diarization algorithm completed but detected 0 distinct speakers.",
            original_exception=e
        )

    # 5. NeMo Manifest & Configuration Errors
    if "omegaconf" in error_msg or "manifest" in error_msg or "json" in error_msg:
        return exc.DiarizationManifestError(
            f"NeMo failed to process the internal configuration or audio manifest: {str(e)}",
            original_exception=e
        )

    # 6. General Embedding / Tensor Errors
    if "tensor" in error_msg or "cuda" in error_msg or "dimension" in error_msg:
        return exc.DiarizationInferenceError(
            f"The NeMo TitaNet model failed during the forward pass: {str(e)}",
            original_exception=e
        )

    # 7. Fallback Catch-all
    return exc.DiarizationError(
        f"An unexpected error occurred in the Diarization feature: {str(e)}",
        original_exception=e
    )