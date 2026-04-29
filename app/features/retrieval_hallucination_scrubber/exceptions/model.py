# Canvas: model_v1
# Path: app/features/retrieval_hallucination_scrubber/exceptions/model.py

from .base import RetrievalScrubberError

class ScrubberModelError(RetrievalScrubberError):
    """Base exception for AI hardware and inference failures in the scrubber."""
    pass

class SLMOOMError(ScrubberModelError):
    """Raised when the GPU VRAM limit is breached during Qwen 2.5 load or inference."""
    pass

class SLMInferenceError(ScrubberModelError):
    """Raised when text/JSON generation fails unexpectedly within the Llama.cpp engine."""
    pass