# Canvas: domain_adaptation_exceptions_init_v1
# Path: app/features/domain_adaptation/exceptions/__init__.py

from .base import DomainAdaptationError
from .database import DomainDatabaseError, GlossaryPersistenceError
from .model import CpuModelLoadError, ScoutInferenceError, ClassifierInferenceError
from .process import AudioSamplingError, GlossaryManifestError, AnchorValidationFailed

__all__ = [
    "DomainAdaptationError",
    "DomainDatabaseError",
    "GlossaryPersistenceError",
    "CpuModelLoadError",
    "ScoutInferenceError",
    "ClassifierInferenceError",
    "AudioSamplingError",
    "GlossaryManifestError",
    "AnchorValidationFailed"
]