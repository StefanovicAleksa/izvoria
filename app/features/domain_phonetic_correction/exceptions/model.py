# Canvas: exceptions_model_v1
# Path: app/features/phonetic_correction/exceptions/model.py

from .base import PhoneticCorrectionError

class PhoneticModelError(PhoneticCorrectionError):
    """Base exception for AI hardware and inference failures."""
    pass

class PhoneticOOMError(PhoneticModelError):
    """Raised when the GPU VRAM limit is breached during Qwen 2.5 load or inference."""
    pass

class GGUFLoadError(PhoneticModelError):
    """Raised when llama-cpp-python fails to load the GGUF weights."""
    pass

class LLMPatchGenerationError(PhoneticModelError):
    """Raised when text/JSON generation fails unexpectedly within the Llama.cpp engine."""
    pass