# Canvas: sentence_error_mapper_v6
# Path: app/features/sentence_reconstruction/utils/error_mapper.py

import torch

from app.core.database import exceptions as db_exc
from app.features.sentence_reconstruction import exceptions as exc


def map_sentence_error(e: Exception) -> exc.SentenceReconstructionError:
    """
    Translates raw system, PyTorch, wtpsplit, and Core DB exceptions into
    highly specific typed Domain Exceptions for Phase 7.
    """
    error_msg = str(e).lower()

    # 1. Catch existing domain errors to prevent double-wrapping
    if isinstance(e, exc.SentenceReconstructionError):
        return e

    # 2. Granular Database Errors (Mapped from the Core DatabaseManager)
    if isinstance(e, db_exc.DBForeignKeyError):
        return exc.SentencePersistenceError("Failed database constraint. Does the Video Source exist?")

    if isinstance(e, (db_exc.DataError, db_exc.DBInvalidTextRepresentation)):
        return exc.SentenceSerializationError(f"JSONB format rejected: {str(e)}")

    if isinstance(e, db_exc.DatabaseError):
        return exc.SentenceDatabaseError(f"Generic database failure: {str(e)}")

    # 3. Model Missing Errors
    if isinstance(e, OSError) and ("model identifier" in error_msg or "not found" in error_msg):
        return exc.SaTModelLoadError(f"SaT model files missing: {str(e)}")

    # 4. CUDA Out Of Memory
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in error_msg:
        return exc.SaTOOMError("GPU out of VRAM. Check ModelManagerOrchestrator cleanup.")

    # 5. Virtual Slicing Math Errors
    if isinstance(e, ZeroDivisionError) or "division by zero" in error_msg:
        return exc.ZeroDurationSegmentError("Cannot virtual-slice a zero-duration segment.")

    # 6. PyTorch/SaT Inference Crashes
    if isinstance(e, RuntimeError) or "tensor" in error_msg or "dimension" in error_msg:
        return exc.SaTExecutionError(f"PyTorch execution failed: {str(e)}")

    # 7. Fallback Catch-all
    return exc.SentenceReconstructionError(f"Unexpected failure: {str(e)}")