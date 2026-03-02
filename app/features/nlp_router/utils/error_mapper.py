# Canvas: nlp_router_error_mapper_v2
# Path: app/features/nlp_router/utils/error_mapper.py

import json
from pydantic import ValidationError

from app.features.nlp_router import exceptions as exc


def map_nlp_router_error(e: Exception, context: str = "") -> exc.NLPRouterError:
    """
    Translates raw system, JSON, Pydantic, and llama.cpp exceptions into
    highly specific typed Domain Exceptions for the NLP Router feature.
    """
    error_msg = str(e).lower()

    # 1. Model Missing
    if isinstance(e, FileNotFoundError):
        return exc.NLPRouterModelLoadError(
            f"The LLM model file could not be found: {str(e)} Context: {context}", # FIXED: Added Context
            original_exception=e
        )

    # 2. JSON Parsing & Hallucination Errors
    if isinstance(e, (json.JSONDecodeError, ValidationError)):
        return exc.NLPRouterJSONParseError(
            f"The LLM hallucinated invalid JSON or violated the schema: {str(e)} Context: {context}",
            original_exception=e
        )

    # 3. Context Window Overflows
    if isinstance(e, ValueError) and ("context" in error_msg or "tokens" in error_msg or "exceed" in error_msg):
        return exc.NLPRouterContextLimitError(
            f"The transcript chunk exceeded the LLM's context window: {str(e)} Context: {context}",
            original_exception=e
        )

    # 4. General Inference / C++ Binding Crashes
    if isinstance(e, RuntimeError) or "llama" in error_msg:
        return exc.NLPRouterInferenceError(
            f"llama.cpp engine failed during inference: {str(e)} Context: {context}",
            original_exception=e
        )

    # 5. Fallback Catch-all
    return exc.NLPRouterError(
        f"An unexpected error occurred in the NLP Router: {str(e)} Context: {context}",
        original_exception=e
    )