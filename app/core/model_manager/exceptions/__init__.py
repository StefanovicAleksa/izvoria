# File: app/core/model_manager/exceptions/__init__.py
from .base import ModelManagerError
from .loading import ModelNotFoundError, ModelLoadError
from .vram import OutOfVramError, CudaCrashError
from .server import LlamaServerError, ResidencyError

__all__ = [
    "ModelManagerError",
    "ModelNotFoundError",
    "ModelLoadError",
    "OutOfVramError",
    "CudaCrashError",
    "LlamaServerError",
    "ResidencyError"
]
