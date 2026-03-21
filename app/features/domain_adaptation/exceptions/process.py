# Canvas: domain_adaptation_exceptions_process_v2
# Path: app/features/domain_adaptation/exceptions/process.py

from .base import DomainAdaptationError

class SmartSamplingError(DomainAdaptationError):
    """Raised when the VadMap is empty, corrupted, or contains no usable speech blocks."""
    pass

class GlossaryManifestError(DomainAdaptationError):
    """Raised when the local master JSON glossaries (e.g., finance.json) are missing or corrupted."""
    pass