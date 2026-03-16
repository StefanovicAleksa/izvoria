# Canvas: domain_adaptation_exceptions_model_v1
# Path: app/features/domain_adaptation/exceptions/model.py

from .base import DomainAdaptationError

class CpuModelLoadError(DomainAdaptationError):
    """Raised when either the Faster-Whisper Scout or DeBERTa Classifier fails to load into RAM."""
    pass

class ScoutInferenceError(DomainAdaptationError):
    """Raised when the CPU Whisper model crashes while trying to transcribe the audio slices."""
    pass

class ClassifierInferenceError(DomainAdaptationError):
    """Raised when the DeBERTa model fails to process the text or rank the candidate labels."""
    pass