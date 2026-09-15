# Path: app/features/qa_reasoning/utils/error_mapper.py

import json
import logging
from pydantic import ValidationError

from app.features.qa_reasoning import exceptions as exc

logger = logging.getLogger(__name__)

def map_reasoner_error(e: Exception, context: str = "") -> exc.QAReasoningError:
    """
    Translates raw system, PyTorch, Llama.cpp, and JSON exceptions into
    highly specific typed Domain Exceptions for the Reasoner.
    """
    if isinstance(e, exc.QAReasoningError):
        return e

    error_msg = str(e).lower()

    # Detect VRAM/Hardware limits. (Using string matching to avoid hard torch dependency if not loaded)
    if "out of memory" in error_msg or "cuda_error_out_of_memory" in error_msg:
        return exc.ReasonerOOMError(
            f"VRAM exhausted during Reasoner inference: {str(e)} | Context: {context}",
            original_exception=e
        )

    # LLM JSON Hallucinations / Strict Schema Failures
    if isinstance(e, (json.JSONDecodeError, ValidationError)) or "json" in error_msg:
        return exc.JSONReasonerParseError(
            f"Reasoner hallucinated invalid JSON or an unrecognized decision: {str(e)} | Context: {context}",
            original_exception=e
        )

    # Llama.cpp Inference Crashes
    if "llama_cpp" in error_msg or "generation" in error_msg or "context window" in error_msg:
        return exc.ReasonerInferenceError(
            f"Reasoner crashed during inference: {str(e)} | Context: {context}",
            original_exception=e
        )

    # Fallback Catch-all
    logger.error(f"Unhandled Reasoner Error: {str(e)}", exc_info=True)
    return exc.QAReasoningError(
        f"Unexpected failure in QA Reasoning: {str(e)} | Context: {context}",
        original_exception=e
    )
