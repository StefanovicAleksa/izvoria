# Canvas: domain_adaptation_exceptions_model_v2
# Path: app/features/domain_adaptation/exceptions/model.py

from .base import DomainAdaptationError

class ModelLoadError(DomainAdaptationError):
    """Raised when the Faster-Whisper Scout or the LLM Router fails to load into VRAM."""
    pass

class ScoutInferenceError(DomainAdaptationError):
    """Raised when the Faster-Whisper model crashes while transcribing the audio slices."""
    pass

class LLMRouterInferenceError(DomainAdaptationError):
    """Raised when the local LLM fails to generate text, OOMs, or outputs invalid JSON."""
    pass

class TokenLimitExceededError(DomainAdaptationError):
    """Raised when the token-compression algorithm fails to keep the hotwords within CTranslate2's limits."""
    pass