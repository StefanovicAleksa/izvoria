# Canvas: domain_adaptation_exceptions_process_v1
# Path: app/features/domain_adaptation/exceptions/process.py

from .base import DomainAdaptationError

class AudioSamplingError(DomainAdaptationError):
    """Raised when FFmpeg fails to extract the 15/50/85% audio slices."""
    pass

class GlossaryManifestError(DomainAdaptationError):
    """Raised when the local master JSON glossary (e.g., finance.json) is missing or corrupted."""
    pass

class AnchorValidationFailed(DomainAdaptationError):
    """
    Raised when the DeBERTa model guesses a sub-domain, but the transcribed text
    contains zero deterministic 'Anchor Words' to prove it. Triggers a fallback.
    """
    pass