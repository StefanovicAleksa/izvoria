# Canvas: sentence_model_exceptions_v3
# Path: app/features/sentence_reconstruction/exceptions/model.py

from .base import SentenceReconstructionError

class SaTModelLoadError(SentenceReconstructionError):
    """Raised when the wtpsplit/SaT model cannot be found on disk or fails to initialize."""
    pass

class SaTOOMError(SentenceReconstructionError):
    """Raised specifically when the GPU runs out of VRAM during sentence splitting."""
    pass

class SaTExecutionError(SentenceReconstructionError):
    """Raised when the model throws a tensor or dimension error during inference."""
    pass