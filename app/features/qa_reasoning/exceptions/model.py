# Path: app/features/qa_reasoning/exceptions/model.py

from .base import QAReasoningError

class ReasonerModelError(QAReasoningError):
    """Base exception for AI hardware and inference failures in the Reasoner."""
    pass

class ReasonerOOMError(ReasonerModelError):
    """Raised when the GPU VRAM limit is breached during Qwen3.5-9B load or inference."""
    pass

class ReasonerInferenceError(ReasonerModelError):
    """Raised when text/JSON generation fails unexpectedly within the Llama.cpp engine."""
    pass
