# Canvas: error_mapper_v1
# Path: app/features/retrieval_hallucination_scrubber/utils/error_mapper.py

import json
import logging
from pydantic import ValidationError

from app.features.retrieval_hallucination_scrubber import exceptions as exc

logger = logging.getLogger(__name__)

def map_scrubber_error(e: Exception, context: str = "") -> exc.RetrievalScrubberError:
    """
    Translates raw system, PyTorch, Llama.cpp, and JSON exceptions into
    highly specific typed Domain Exceptions for the Scrubber phase.
    """
    error_msg = str(e).lower()

    if isinstance(e, exc.RetrievalScrubberError):
        return e

    # Detect VRAM/Hardware limits. (Using string matching to avoid hard torch dependency if not loaded)
    if "out of memory" in error_msg or "cuda_error_out_of_memory" in error_msg:
        return exc.SLMOOMError(
            f"VRAM exhausted during JSON relevance audit: {str(e)} | Context: {context}",
            original_exception=e
        )

    # LLM JSON Hallucinations / Strict Schema Failures
    if isinstance(e, (json.JSONDecodeError, ValidationError)) or "json" in error_msg:
        return exc.JSONScrubParseError(
            f"SLM hallucinated invalid JSON during the relevance audit: {str(e)} | Context: {context}",
            original_exception=e
        )

    # Llama.cpp Inference Crashes
    if "llama_cpp" in error_msg or "generation" in error_msg or "context window" in error_msg:
        return exc.SLMInferenceError(
            f"SLM crashed during inference evaluation: {str(e)} | Context: {context}",
            original_exception=e
        )

    # Fallback Catch-all
    logger.error(f"Unhandled Scrubber Error: {str(e)}", exc_info=True)
    return exc.RetrievalScrubberError(
        f"Unexpected failure in Retrieval Scrubber: {str(e)} | Context: {context}",
        original_exception=e
    )