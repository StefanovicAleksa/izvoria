# Canvas: qa_generation_error_mapper_v1
# Path: app/features/qa_generation/utils/error_mapper.py

import logging
from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.qa_generation import exceptions as exc

logger = logging.getLogger(__name__)

def map_qa_error(e: Exception, context: str = "") -> exc.QAGenerationError:
    """
    Translates raw system, PyTorch, Llama.cpp, and core ModelManager exceptions
    into highly specific typed Domain Exceptions for the QA Generation phase.
    """
    if isinstance(e, exc.QAGenerationError):
        return e

    error_msg = str(e).lower()

    if "out of memory" in error_msg or "cuda_error_out_of_memory" in error_msg:
        return exc.GemmaOOMError(
            f"VRAM exhausted during QA Generation inference: {str(e)} | Context: {context}",
            original_exception=e
        )

    if "token indices sequence length" in error_msg or "context length" in error_msg or "n_ctx" in error_msg:
        return exc.ContextWindowExceededError(
            f"The combined RAG context and user prompt exceed Gemma's context window: {str(e)} | Context: {context}",
            original_exception=e
        )

    if "llama_cpp" in error_msg or "generation failed" in error_msg or "segmentation fault" in error_msg:
        return exc.LLMInferenceCrashError(
            f"Gemma model crashed during text generation: {str(e)} | Context: {context}",
            original_exception=e
        )

    if isinstance(e, ModelManagerError):
        return exc.GenerationModelError(
            f"Model Manager failed to provision the Gemma model: {str(e)} | Context: {context}",
            original_exception=e
        )

    logger.error(f"Unhandled QA Generation Error: {str(e)}", exc_info=True)
    return exc.QAGenerationError(
        f"An unexpected failure occurred during QA Generation: {str(e)} | Context: {context}",
        original_exception=e
    )