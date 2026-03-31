# Canvas: visual_analysis_exceptions_model_v1
# Path: app/features/visual_analysis/exceptions/model.py

from .base import VisualAnalysisError

class VLMModelError(VisualAnalysisError):
    """Base exception for AI hardware and VLM inference failures."""
    pass

class VLMInferenceError(VLMModelError):
    """Raised when the Vision-Language Model crashes, hallucinates badly, or the GPU OOMs."""
    pass

class ImageDecodeError(VisualAnalysisError):
    """Raised when the physical JPEG is missing from the Vault or is corrupted."""
    pass