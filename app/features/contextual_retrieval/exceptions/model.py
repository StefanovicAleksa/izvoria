# Canvas: contextual_exceptions_model_v1
# Path: app/features/contextual_retrieval/exceptions/model.py

from .base import ContextualRetrievalError

class ContextualModelError(ContextualRetrievalError):
    """Base exception for AI hardware and SLM inference failures."""
    pass

class SLMModelLoadError(ContextualModelError):
    """Raised when the SLM weights (e.g., Llama 3.2 1B/3B GGUF) cannot be found or loaded."""
    pass

class SLMInferenceError(ContextualModelError):
    """Raised when the SLM crashes or hallucinates during context generation, or if the GPU OOMs."""
    pass

class ContextWindowExceededError(ContextualModelError):
    """Raised when the chunk + document exceeds the maximum token length allowed by the SLM."""
    pass