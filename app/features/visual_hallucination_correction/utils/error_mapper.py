# Canvas: visual_hallucination_error_mapper_v1
# Path: app/features/visual_hallucination_correction/utils/error_mapper.py

import json
import torch
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

from app.core.database import exceptions as core_db_exc
from app.features.visual_hallucination_correction import exceptions as exc


def map_hallucination_error(e: Exception, context: str = "") -> exc.VisualHallucinationError:
    """
    Translates raw system, PyTorch, Llama.cpp, JSON, and Database exceptions into
    highly specific typed Domain Exceptions for the Visual Hallucination Correction phase.
    """
    error_msg = str(e).lower()

    # 1. Prevent double-wrapping if it is already a domain error
    if isinstance(e, exc.VisualHallucinationError):
        return e

    # 2. VRAM / Hardware Limits
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in error_msg:
        return exc.SLMOOMError(
            f"VRAM exhausted during JSON visual audit generation: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. LLM JSON Hallucinations / Strict Schema Failures
    if isinstance(e, (json.JSONDecodeError, ValidationError)) or "json" in error_msg:
        return exc.JSONAuditParseError(
            f"SLM hallucinated invalid JSON during the visual audit: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. Llama.cpp Inference Crashes
    if "llama_cpp" in error_msg or "generation" in error_msg or "context window" in error_msg:
        return exc.SLMInferenceError(
            f"SLM crashed during inference evaluation: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 5. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        if "audit" in context.lower() or "transcript" in context.lower():
            return exc.AuditPersistenceError(
                f"Failed to save evaluated transcript and audit ledger: {str(e)}",
                original_exception=e
            )
        return exc.VisualHallucinationDatabaseError(
            f"Database operation failed in Visual Hallucination Correction: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 6. Fallback Catch-all
    return exc.VisualHallucinationError(
        f"Unexpected failure in Visual Hallucination Correction: {str(e)} | Context: {context}",
        original_exception=e
    )