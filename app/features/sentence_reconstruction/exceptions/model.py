# Canvas: sentence_exceptions_model_v4
# Path: app/features/sentence_reconstruction/exceptions/model.py

from .base import SentenceReconstructionError

class PunctuationModelLoadError(SentenceReconstructionError):
    """Raised when the fullstop token classifier cannot be found on disk or fails to initialize."""
    pass

class PunctuationOOMError(SentenceReconstructionError):
    """Raised specifically when the GPU runs out of VRAM during punctuation tagging."""
    pass

class PunctuationExecutionError(SentenceReconstructionError):
    """Raised when the Hugging Face pipeline throws a tensor or dimension error during inference."""
    pass