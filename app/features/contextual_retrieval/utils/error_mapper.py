# Canvas: contextual_error_mapper_v1
# Path: app/features/contextual_retrieval/utils/error_mapper.py

from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.contextual_retrieval import exceptions as exc


def map_contextual_error(e: Exception, context: str = "") -> exc.ContextualRetrievalError:
    """
    Translates raw system, Llama.cpp, memory, and DB exceptions into
    highly specific typed Domain Exceptions for the Contextual Retrieval feature.
    """
    error_msg = str(e).lower()

    # 1. Prevent double-wrapping if it's already a domain error
    if isinstance(e, exc.ContextualRetrievalError):
        return e

    # 2. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        if "transcript" in context.lower() or "chunk" in context.lower():
            return exc.ContextualPersistenceError(
                f"Failed to save Contextualized Transcript: {str(e)}",
                original_exception=e
            )
        return exc.ContextualDatabaseError(
            f"Database operation failed in Contextual Retrieval: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. Model Loading Errors (Llama.cpp GGUF)
    if isinstance(e, FileNotFoundError) or "gguf" in error_msg or "load model" in error_msg:
        return exc.SLMModelLoadError(
            f"Failed to load GGUF weights for SLM Context Adapter: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. Context Window Overflow
    if "context window" in error_msg or "maximum context" in error_msg or "n_ctx" in error_msg:
        return exc.ContextWindowExceededError(
            f"The document sliding window exceeded the SLM's max token limit: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 5. VRAM / Inference / Llama.cpp crashes
    if "out of memory" in error_msg or "llama_cpp" in error_msg or "generation" in error_msg or "tensor" in error_msg:
        return exc.SLMInferenceError(
            f"SLM crashed during context generation. GPU VRAM might be exhausted: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 6. Python Logic / Chunking bounds
    if isinstance(e, (ValueError, IndexError)) and "chunk" in context.lower():
        return exc.TranscriptChunkingError(
            f"Failed to cleanly chunk the transcript sentences: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 7. Fallback Catch-all
    return exc.ContextualRetrievalError(
        f"Unexpected failure in Contextual Retrieval: {str(e)} | Context: {context}",
        original_exception=e
    )