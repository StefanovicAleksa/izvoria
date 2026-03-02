# Canvas: nlp_router_exceptions_model_v1
# Path: app/features/nlp_router/exceptions/model.py

from app.features.nlp_router.exceptions.base import NLPRouterError

class NLPRouterModelLoadError(NLPRouterError):
    """Raised when the GGUF model file is missing or corrupted."""
    pass

class NLPRouterInferenceError(NLPRouterError):
    """Raised when the llama.cpp C-bindings crash during text generation."""
    pass

class NLPRouterJSONParseError(NLPRouterError):
    """
    Raised when the LLM hallucinates and fails to return valid JSON,
    or when the JSON does not match our strict Pydantic schema.
    """
    pass

class NLPRouterContextLimitError(NLPRouterError):
    """
    Raised when the provided transcript chunk exceeds the maximum
    token context window (e.g., 4096 tokens for Phi-3.5).
    """
    pass