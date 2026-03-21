# Canvas: domain_adaptation_exceptions_system_v2
# Path: app/features/domain_adaptation/exceptions/system.py

from .base import DomainAdaptationError

class AudioMemoryProcessingError(DomainAdaptationError):
    """Raised when torchaudio fails to load, resample, or chunk the master audio file in RAM."""
    pass