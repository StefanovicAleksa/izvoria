# Canvas: sentence_error_mapper_v8
# Path: app/features/sentence_reconstruction/utils/error_mapper.py

import torch
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as db_exc
from app.features.sentence_reconstruction import exceptions as exc


def map_sentence_error(e: Exception) -> exc.SentenceReconstructionError:
    """
    Translates raw system, PyTorch, Hugging Face, and Core DB exceptions into
    highly specific typed Domain Exceptions for Phase 7.
    """
    error_msg = str(e).lower()

    # 1. Catch existing domain errors to prevent double-wrapping
    if isinstance(e, exc.SentenceReconstructionError):
        return e

    # 2. Granular Database Errors (Mapped from the Core DatabaseManager)
    if isinstance(e, db_exc.DBForeignKeyError):
        return exc.SentencePersistenceError(
            "Failed database constraint. Does the Video Source exist?",
            original_exception=e
        )

    if isinstance(e, (db_exc.DataError, db_exc.DBInvalidTextRepresentation)):
        return exc.SentenceSerializationError(
            f"JSONB format rejected: {str(e)}",
            original_exception=e
        )

    if isinstance(e, (db_exc.DatabaseError, SQLAlchemyError)):
        return exc.SentenceDatabaseError(
            f"Generic database failure: {str(e)}",
            original_exception=e
        )

    # 3. Model Loading Errors (Hugging Face / Transformers)
    # CRITICAL FIX: Also check for "missing"
    if isinstance(e, (OSError, EnvironmentError)) and ("model identifier" in error_msg or "not found" in error_msg or "missing" in error_msg):
        return exc.PunctuationModelLoadError(
            f"Punctuation model files missing or corrupted: {str(e)}",
            original_exception=e
        )

    # 4. CUDA Out Of Memory
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in error_msg:
        return exc.PunctuationOOMError(
            "GPU out of VRAM during punctuation tagging. Check ModelManagerOrchestrator cleanup.", original_exception=e
        )

    # 5. PyTorch / Transformers Inference Crashes
    if isinstance(e, RuntimeError) or "tensor" in error_msg or "dimension" in error_msg or "pipeline" in error_msg:
        return exc.PunctuationExecutionError(
            f"Hugging Face pipeline execution failed: {str(e)}",
            original_exception=e
        )

    # 6. Fallback Catch-all
    return exc.SentenceReconstructionError(
        f"Unexpected failure in Sentence Reconstruction: {str(e)}",
        original_exception=e
    )