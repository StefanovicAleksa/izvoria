# filename: error_mapper_v2.py
# path: app/core/model_manager/utils/error_mapper.py

from app.core.model_manager import exceptions as exc

def map_model_manager_error(e: Exception) -> exc.ModelManagerError:
    """
    Translates raw system, PyTorch, and CUDA exceptions into
    typed Domain Exceptions for the Model Manager.
    """
    # --- NEW GUARD CLAUSE ---
    # If the exception is already a Domain Exception (like OutOfVramError
    # raised by the Orchestrator), return it as-is to prevent double-wrapping.
    if isinstance(e, exc.ModelManagerError):
        return e

    error_msg = str(e).lower()

    # 1. Hardware / CUDA Out Of Memory (OOM)
    # PyTorch raises a generic RuntimeError for OOM, so we must inspect the string
    if "cuda out of memory" in error_msg or isinstance(e, MemoryError):
        return exc.OutOfVramError(
            "Hardware constraint reached: Not enough VRAM available to load the model.",
            original_exception=e
        )

    # 2. CUDA Runtime Crashes (Device lost, driver issues)
    if "cuda" in error_msg and ("device-side assert" in error_msg or "invalid device" in error_msg):
        return exc.CudaCrashError(
            "The NVIDIA GPU encountered a fatal runtime error.",
            original_exception=e
        )

    # 3. Missing Files
    if isinstance(e, FileNotFoundError):
        return exc.ModelNotFoundError(
            "The requested model weight file could not be found on disk.",
            original_exception=e
        )

    # 4. Corrupted Weights / Deserialization Failures
    # PyTorch raises EOFError, RuntimeError, or Pickle errors for bad weights
    if isinstance(e, (EOFError, TypeError)) or "invalid load key" in error_msg or "magic number" in error_msg:
        return exc.ModelLoadError(
            "Failed to load model weights. The file may be corrupted or incompatible.",
            original_exception=e
        )

    # 5. Fallback Catch-all
    return exc.ModelManagerError(
        f"An unexpected error occurred in the Model Manager: {str(e)}",
        original_exception=e
    )