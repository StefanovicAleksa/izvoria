# Canvas: visual_hallucination_exceptions_model_v1
# Path: app/features/visual_hallucination_correction/exceptions/model.py

from .base import VisualHallucinationError

class HallucinationModelError(VisualHallucinationError):
    """Base exception for AI hardware and inference failures in Phase 13."""
    pass

class SLMOOMError(HallucinationModelError):
    """Raised when the GPU VRAM limit is breached during Qwen 2.5 load or inference."""
    pass

class SLMInferenceError(HallucinationModelError):
    """Raised when text/JSON generation fails unexpectedly within the Llama.cpp engine."""
    pass