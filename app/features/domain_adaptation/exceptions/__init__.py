# Canvas: domain_adaptation_exceptions_init_v3
# Path: app/features/domain_adaptation/exceptions/__init__.py

from .base import DomainAdaptationError
from .database import DomainDatabaseError, GlossaryPersistenceError
from .system import AudioMemoryProcessingError
from .model import (
    ModelLoadError,
    ScoutInferenceError,
    LLMRouterInferenceError,
    TokenLimitExceededError
)
from .process import SmartSamplingError, GlossaryManifestError

__all__ = [
    "DomainAdaptationError",
    "DomainDatabaseError",
    "GlossaryPersistenceError",
    "AudioMemoryProcessingError",
    "ModelLoadError",
    "ScoutInferenceError",
    "LLMRouterInferenceError",
    "TokenLimitExceededError",
    "SmartSamplingError",
    "GlossaryManifestError"
]