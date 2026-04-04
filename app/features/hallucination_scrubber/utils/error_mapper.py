# Canvas: scrubber_error_mapper_v1
# Path: app/features/hallucination_scrubber/utils/error_mapper.py

import torch
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.hallucination_scrubber import exceptions as exc


def map_scrubber_error(e: Exception, context: str = "") -> exc.HallucinationScrubberError:
    """
    Translates raw system, PyTorch, HuggingFace, and DB exceptions into
    highly specific typed Domain Exceptions for the Hallucination Scrubber feature.
    """
    error_msg = str(e).lower()

    # 1. Prevent double-wrapping if it's already a domain error
    if isinstance(e, exc.HallucinationScrubberError):
        return e

    # 2. VRAM / Hardware Limits
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in error_msg:
        return exc.ScrubberOOMError(
            f"VRAM exhausted during Zero-Shot classification. Ensure previous models unloaded: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. Model Loading Errors (HuggingFace Transformers)
    if isinstance(e, (FileNotFoundError, OSError)) or "model path" in error_msg or "config.json" in error_msg:
        return exc.ZeroShotLoadError(
            f"Failed to load DeBERTa weights or tokenizer for Scrubber Adapter: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. HuggingFace / PyTorch Inference Crashes
    if "pipeline" in error_msg or "tensor" in error_msg or isinstance(e, RuntimeError):
        return exc.DebertaInferenceError(
            f"DeBERTa model failed during classification: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 5. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        if "transcript" in context.lower() or "audit" in context.lower():
            return exc.ScrubbedTranscriptSaveError(
                f"Failed to save scrubbed transcript and audit trail: {str(e)}",
                original_exception=e
            )
        return exc.ScrubbingDatabaseError(
            f"Database operation failed in Hallucination Scrubber: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 6. Pipeline / Text Application Errors
    if isinstance(e, (ValueError, IndexError)) and "clause" in context.lower():
        return exc.TextSegmentationError(
            f"Failed to segment text into clauses for batch evaluation: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 7. Fallback Catch-all
    return exc.HallucinationScrubberError(
        f"Unexpected failure in Hallucination Scrubber: {str(e)} | Context: {context}",
        original_exception=e
    )