# Canvas: visual_synthesis_exceptions_model_v1
# Path: app/features/visual_context_synthesis/exceptions/model.py

from .base import VisualSynthesisError

class VisualSynthesisModelError(VisualSynthesisError):
    """Base exception for AI hardware and SLM inference failures during synthesis."""
    pass

class VisualSynthesisOOMError(VisualSynthesisModelError):
    """Raised when the GPU VRAM limit is breached while attempting to load the Llama 3.2 SLM."""
    pass

class VisualSLMLoadError(VisualSynthesisModelError):
    """Raised when the SLM weights (GGUF) cannot be found, are corrupted, or fail to mount in llama.cpp."""
    pass

class VisualSLMInferenceError(VisualSynthesisModelError):
    """Raised when the SLM crashes, hallucinates, or deadlocks during context generation."""
    pass