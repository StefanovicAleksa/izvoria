# Canvas: scrubber_exceptions_model_v1
# Path: app/features/hallucination_scrubber/exceptions/model.py

from .base import HallucinationScrubberError

class ScrubberModelError(HallucinationScrubberError):
    """Base exception for AI hardware and Zero-Shot inference failures."""
    pass

class ScrubberOOMError(ScrubberModelError):
    """Raised when the GPU VRAM limit is breached while loading DeBERTa or processing a massive batch."""
    pass

class ZeroShotLoadError(ScrubberModelError):
    """Raised when HuggingFace Transformers fails to load the model weights or tokenizer."""
    pass

class DebertaInferenceError(ScrubberModelError):
    """Raised when the DeBERTa model crashes unexpectedly during the classification sequence."""
    pass