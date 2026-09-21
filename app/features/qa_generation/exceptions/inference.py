# Canvas: qa_generation_exceptions_inference_v1
# Path: app/features/qa_generation/exceptions/inference.py

from .base import QAGenerationError

class GenerationModelError(QAGenerationError):
    """Base exception for AI hardware and inference failures during QA generation."""
    pass

class GenerationOOMError(GenerationModelError):
    """
    Raised when the GPU VRAM limit is breached while loading the QA generation model
    or during the autoregressive generation phase.
    """
    pass

class ContextWindowExceededError(GenerationModelError):
    """
    Raised when the assembled RAG context chunks + user prompt exceed
    the maximum configured token context limit of the QA generation model.
    """
    pass

class LLMInferenceCrashError(GenerationModelError):
    """
    Raised when the generation engine crashes unexpectedly
    mid-generation due to internal faults or hardware disconnects.
    """
    pass