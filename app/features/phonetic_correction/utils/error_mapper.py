# Canvas: phonetic_error_mapper_v1
# Path: app/features/phonetic_correction/utils/error_mapper.py

import json
import torch
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

from app.core.database import exceptions as core_db_exc
from app.features.phonetic_correction import exceptions as exc


def map_phonetic_error(e: Exception, context: str = "") -> exc.PhoneticCorrectionError:
    """
    Translates raw system, PyTorch, Llama.cpp, and JSON exceptions into
    highly specific typed Domain Exceptions for the Phonetic Correction feature.
    """
    error_msg = str(e).lower()

    # 1. Prevent double-wrapping if it's already a domain error
    if isinstance(e, exc.PhoneticCorrectionError):
        return e

    # 2. VRAM / Hardware Limits
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in error_msg:
        return exc.PhoneticOOMError(
            f"VRAM exhausted during JSON Patch generation. Ensure previous models unloaded: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. Model Loading Errors (Llama.cpp GGUF)
    if isinstance(e, FileNotFoundError) or "gguf" in error_msg or "load model" in error_msg:
        return exc.GGUFLoadError(
            f"Failed to load GGUF weights for Phonetic Adapter: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. LLM JSON Hallucinations / Formatting Failures
    if isinstance(e, (json.JSONDecodeError, ValidationError)):
        return exc.JSONPatchParseError(
            f"Qwen model hallucinated invalid JSON during patch generation: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 5. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        if "transcript" in context.lower() or "audit" in context.lower():
            return exc.TranscriptSaveError(
                f"Failed to save corrected transcript and audit trail: {str(e)}",
                original_exception=e
            )
        return exc.PhoneticDatabaseError(
            f"Database operation failed in Phonetic Correction: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 6. Llama.cpp Inference Crashes
    if "llama_cpp" in error_msg or "generation" in error_msg:
        return exc.LLMPatchGenerationError(
            f"Qwen model failed during text/JSON generation: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 7. Fallback Catch-all
    return exc.PhoneticCorrectionError(
        f"Unexpected failure in Phonetic Correction: {str(e)} | Context: {context}",
        original_exception=e
    )