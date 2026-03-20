from .base import AsrCorrectionError

class AsrCorrectionModelError(AsrCorrectionError):
    """Base exception for all AI model failures in this feature."""
    pass

class AsrCorrectionOOMError(AsrCorrectionModelError):
    """
    Raised when the 12GB VRAM limit is breached.
    Crucial for catching situations where a model wasn't properly unloaded
    before the next phase began.
    """
    pass

class GGUFModelLoadError(AsrCorrectionModelError):
    """Raised when llama-cpp-python fails to load the Llama 3.2 or Qwen 2.5 weights."""
    pass

class T5InferenceError(AsrCorrectionModelError):
    """Raised when the native PyTorch T5 anomaly detection model crashes during the forward pass."""
    pass

class LLMInferenceError(AsrCorrectionModelError):
    """Raised when text generation fails in Llama 3.2 or Qwen 2.5."""
    pass