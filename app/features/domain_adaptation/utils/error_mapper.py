# Canvas: domain_adaptation_error_mapper_v1
# Path: app/features/domain_adaptation/utils/error_mapper.py

import json
import subprocess
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.domain_adaptation import exceptions as exc


def map_domain_adaptation_error(e: Exception, context: str = "") -> exc.DomainAdaptationError:
    """
    Translates raw system, FFmpeg, HuggingFace, CTranslate2, and DB exceptions into
    typed Domain Exceptions for the Domain Adaptation feature.
    """
    error_msg = str(e).lower()

    # 1. Guard Clause: Don't double-wrap existing domain errors
    if isinstance(e, exc.DomainAdaptationError):
        return e

    # 2. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        return exc.DomainDatabaseError(
            f"Database operation failed in Domain Adaptation: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. Audio Slicing / FFmpeg Errors
    if isinstance(e, subprocess.CalledProcessError) or isinstance(e, subprocess.TimeoutExpired):
        return exc.AudioSamplingError(
            f"FFmpeg failed to extract audio samples for the Scout: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. JSON Glossary Manifest Errors
    if isinstance(e, json.JSONDecodeError):
        return exc.GlossaryManifestError(
            f"The local industry glossary JSON is corrupted: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 5. Missing Files (Models or Glossaries)
    if isinstance(e, FileNotFoundError):
        if "json" in error_msg:
            return exc.GlossaryManifestError(
                f"Required glossary file missing: {str(e)} | Context: {context}",
                original_exception=e
            )
        return exc.CpuModelLoadError(
            f"Required CPU model weights missing from disk: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 6. HuggingFace / CTranslate2 Inference Crashes
    if isinstance(e, RuntimeError) or "tensor" in error_msg or "ctranslate2" in error_msg:
        if "whisper" in context.lower():
            return exc.ScoutInferenceError(
                f"CPU Whisper Scout crashed during transcription: {str(e)} | Context: {context}",
                original_exception=e
            )
        else:
            return exc.ClassifierInferenceError(
                f"DeBERTa Classifier crashed during zero-shot evaluation: {str(e)} | Context: {context}",
                original_exception=e
            )

    # 7. Fallback Catch-all
    return exc.DomainAdaptationError(
        f"An unexpected error occurred in Domain Adaptation: {str(e)} | Context: {context}",
        original_exception=e
    )