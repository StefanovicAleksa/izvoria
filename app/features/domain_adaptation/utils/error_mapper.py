# Canvas: domain_adaptation_error_mapper_v3
# Path: app/features/domain_adaptation/utils/error_mapper.py

import json
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.domain_adaptation import exceptions as exc


def map_domain_adaptation_error(e: Exception, context: str = "") -> exc.DomainAdaptationError:
    """
    Translates raw system, PyTorch, CTranslate2, LLM, and DB exceptions into
    typed Domain Exceptions for the Domain Adaptation feature.
    """
    error_msg = str(e).lower()

    if isinstance(e, exc.DomainAdaptationError):
        return e

    # 1. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        return exc.DomainDatabaseError(
            f"Database operation failed in Domain Adaptation: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 2. Audio Memory / RAM Slicing Errors
    if "audio" in error_msg or "torchaudio" in error_msg or isinstance(e, (ValueError, IndexError)):
        if "scout" in context.lower() or "slice" in context.lower():
            return exc.AudioMemoryProcessingError(
                f"Failed to process or chunk the audio in RAM: {str(e)} | Context: {context}",
                original_exception=e
            )

    # 3. JSON Glossary / Missing File Errors
    if isinstance(e, json.JSONDecodeError):
        if "llm" in context.lower() or "router" in context.lower() or "llama" in context.lower():
            return exc.LLMRouterInferenceError(
                f"LLM generated invalid JSON that could not be parsed: {str(e)} | Context: {context}",
                original_exception=e
            )
        return exc.GlossaryManifestError(
            f"The local industry glossary JSON is corrupted: {str(e)} | Context: {context}",
            original_exception=e
        )
    if isinstance(e, FileNotFoundError):
        if "json" in error_msg:
            return exc.GlossaryManifestError(
                f"Required glossary file missing: {str(e)} | Context: {context}",
                original_exception=e
            )
        return exc.ModelLoadError(
            f"Required model weights missing from disk: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. Model Inference Crashes (GPU/VRAM issues)
    if isinstance(e, RuntimeError) or "tensor" in error_msg or "ctranslate2" in error_msg or "cuda" in error_msg:
        if "whisper" in context.lower() or "scout" in context.lower():
            return exc.ScoutInferenceError(
                f"GPU Whisper Scout crashed during transcription: {str(e)} | Context: {context}",
                original_exception=e
            )
        else:
            return exc.LLMRouterInferenceError(
                f"Llama Router crashed during domain prediction: {str(e)} | Context: {context}",
                original_exception=e
            )

    # 5. Fallback Catch-all
    return exc.DomainAdaptationError(
        f"An unexpected error occurred in Domain Adaptation: {str(e)} | Context: {context}",
        original_exception=e
    )