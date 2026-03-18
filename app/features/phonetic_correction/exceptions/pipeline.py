# Canvas: exceptions_pipeline_v1
# Path: app/features/phonetic_correction/exceptions/pipeline.py

from .base import PhoneticCorrectionError

class PhoneticPipelineError(PhoneticCorrectionError):
    """Base exception for parsing and data-flow errors."""
    pass

class JSONPatchParseError(PhoneticPipelineError):
    """Raised when Qwen 2.5 hallucinates invalid JSON instead of the strict read-only patch array."""
    pass

class PatchApplicationError(PhoneticPipelineError):
    """Raised when a successfully parsed JSON patch fails to apply to the transcript safely."""
    pass