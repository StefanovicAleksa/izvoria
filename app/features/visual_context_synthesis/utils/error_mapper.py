# Canvas: visual_synthesis_error_mapper_v1
# Path: app/features/visual_context_synthesis/utils/error_mapper.py

from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

from app.core.database import exceptions as core_db_exc
from app.features.visual_context_synthesis import exceptions as exc


def map_visual_synthesis_error(e: Exception, context: str = "") -> exc.VisualSynthesisError:
    """
    Translates raw system, Llama.cpp, memory, and Database exceptions into
    highly specific typed Domain Exceptions for the Visual Context Synthesis feature.
    """
    error_msg = str(e).lower()

    # 1. Prevent double-wrapping if it's already a domain error
    if isinstance(e, exc.VisualSynthesisError):
        return e

    # 2. VRAM / Hardware Limits
    if "out of memory" in error_msg or "memoryerror" in error_msg:
        return exc.VisualSynthesisOOMError(
            f"VRAM exhausted during Visual Context Synthesis. SLM failed to load: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. Model Loading Errors (Llama.cpp GGUF)
    if isinstance(e, FileNotFoundError) or "gguf" in error_msg or "load model" in error_msg:
        return exc.VisualSLMLoadError(
            f"Failed to load GGUF weights for Visual Synthesis SLM Adapter: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. Llama.cpp Inference Crashes
    if "llama_cpp" in error_msg or "generation" in error_msg or "context window" in error_msg or "tensor" in error_msg:
        return exc.VisualSLMInferenceError(
            f"SLM crashed during visual synthesis inference: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 5. Math / Fuzzy Grouping Failures
    # Traps index out of bounds, type errors, or math domain errors during Jaccard calculation
    if isinstance(e, (ValueError, IndexError, TypeError)) and ("fuzzy" in context.lower() or "grouping" in context.lower()):
        return exc.FuzzyGroupingError(
            f"Failed to mathematically group visual time-spans: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 6. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        if "timeline" in context.lower() or "save" in context.lower():
            return exc.VisualTimelinePersistenceError(
                f"Failed to serialize and save Visual Timeline to PostgreSQL: {str(e)}",
                original_exception=e
            )
        return exc.VisualSynthesisDatabaseError(
            f"Database operation failed in Visual Context Synthesis: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 7. Data Validation Errors
    if isinstance(e, ValidationError):
        return exc.VisualSynthesisPipelineError(
            f"Pydantic validation failed while constructing Visual Timeline entities: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 8. Fallback Catch-all
    return exc.VisualSynthesisError(
        f"Unexpected failure in Visual Context Synthesis: {str(e)} | Context: {context}",
        original_exception=e
    )