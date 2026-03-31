# Canvas: visual_analysis_error_mapper_v1
# Path: app/features/visual_analysis/utils/error_mapper.py

from sqlalchemy.exc import SQLAlchemyError
from PIL import UnidentifiedImageError

from app.core.database import exceptions as core_db_exc
from app.core.model_manager import exceptions as core_model_exc
from app.features.visual_analysis import exceptions as exc


def map_visual_analysis_error(e: Exception, context: str = "") -> exc.VisualAnalysisError:
    """
    Translates raw system, Llama.cpp, Image I/O, and DB exceptions into
    strictly typed Domain Exceptions for the Visual Analysis feature.
    """
    error_msg = str(e).lower()

    # 1. Prevent double-wrapping if it's already our domain error
    if isinstance(e, exc.VisualAnalysisError):
        return e

    # 2. Let Core Model Manager errors bubble up (like OutOfVramError)
    # The Orchestrator handles these specifically, so we don't want to swallow them.
    if isinstance(e, core_model_exc.ModelManagerError):
        raise e

    # 3. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        if "chunk" in context.lower() or "analysis" in context.lower():
            return exc.VisualPersistenceError(
                f"Failed to save ChunkVisualAnalysis mapping: {str(e)} | Context: {context}",
                original_exception=e
            )
        return exc.VisualDatabaseError(
            f"Database operation failed in Visual Analysis: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. Image File / Decoding Errors
    if isinstance(e, FileNotFoundError) or isinstance(e, UnidentifiedImageError) or "image" in error_msg:
        return exc.ImageDecodeError(
            f"Failed to read or decode physical frame from Storage Vault: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 5. VLM Inference / Llama.cpp crashes
    if "llama_cpp" in error_msg or "generation" in error_msg or "tensor" in error_msg or "multimodal" in error_msg:
        return exc.VLMInferenceError(
            f"VLM crashed during multimodal inference: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 6. Fallback Catch-all
    return exc.VisualAnalysisError(
        f"Unexpected failure in Visual Analysis: {str(e)} | Context: {context}",
        original_exception=e
    )