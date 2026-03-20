import json
import torch
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

from app.core.database import exceptions as core_db_exc
from app.legacy_features.asr_error_correction import exceptions as exc


def map_asr_correction_error(e: Exception, context: str = "") -> exc.AsrCorrectionError:
    """
    Translates raw system, PyTorch, llama.cpp, and JSON exceptions into
    highly specific typed Domain Exceptions for the ASR Error Correction feature.
    """
    error_msg = str(e).lower()

    # 1. Catch existing domain errors to prevent double-wrapping
    if isinstance(e, exc.AsrCorrectionError):
        return e

    # 2. VRAM / Hardware Limits (Crucial for our 12GB constraint)
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in error_msg:
        return exc.AsrCorrectionOOMError(
            f"VRAM exhausted during 3-pass correction. Ensure sequential unloading: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. Model Loading Errors (llama.cpp GGUF and PyTorch)
    if isinstance(e, FileNotFoundError) or "gguf" in error_msg or "load model" in error_msg:
        return exc.GGUFModelLoadError(
            f"Failed to load GGUF weights. File missing or corrupted: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. LLM JSON Hallucinations
    if isinstance(e, (json.JSONDecodeError, ValidationError)):
        return exc.GlossaryJSONParseError(
            f"LLM hallucinated invalid JSON during Glossary extraction: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 5. Database Errors (Mapped from Core)
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        if "glossary" in context.lower():
            return exc.GlossaryPersistenceError(f"Failed to save Glossary: {str(e)}", original_exception=e)
        if "audit" in context.lower():
            return exc.AuditTrailSaveError(f"Failed to save Audit Trail: {str(e)}", original_exception=e)

        return exc.AsrCorrectionDatabaseError(
            f"Database operation failed in ASR Correction: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 6. PyTorch T5 Inference Crashes
    if isinstance(e, RuntimeError) and ("tensor" in error_msg or "dimension" in error_msg):
        return exc.T5InferenceError(
            f"T5-ASR-EC model failed during forward pass: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 7. llama.cpp Inference Crashes
    if "llama_cpp" in error_msg or "generation" in error_msg:
        return exc.LLMInferenceError(
            f"GGUF model failed during text generation: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 8. Fallback Catch-all
    return exc.AsrCorrectionError(
        f"Unexpected failure in ASR Error Correction: {str(e)} | Context: {context}",
        original_exception=e
    )