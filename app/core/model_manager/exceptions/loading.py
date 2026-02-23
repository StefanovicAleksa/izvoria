# File: app/core/model_manager/exceptions/loading.py
from app.core.model_manager.exceptions.base import ModelManagerError

class ModelNotFoundError(ModelManagerError):
    """Raised when the requested model weight file does not exist in the local vault."""
    pass

class ModelLoadError(ModelManagerError):
    """
    Raised when the weights are found, but fail to load into VRAM
    (e.g., corrupted file, unsupported serialization format, CUDA version mismatch).
    """
    pass