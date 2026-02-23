# File: app/core/model_manager/exceptions/vram.py
from app.core.model_manager.exceptions.base import ModelManagerError

class OutOfVramError(ModelManagerError):
    """
    Raised when the Orchestrator calculates that loading a model
    would exceed the available 12GB GPU memory.
    """
    pass

class CudaCrashError(ModelManagerError):
    """
    Raised when the underlying NVIDIA CUDA runtime panics,
    loses the device connection, or encounters a hardware-level fault.
    """
    pass