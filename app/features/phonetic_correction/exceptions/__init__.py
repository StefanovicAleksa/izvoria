# Canvas: exceptions_init_v1
# Path: app/features/phonetic_correction/exceptions/__init__.py

from .base import PhoneticCorrectionError
from .database import PhoneticDatabaseError, TranscriptSaveError
from .model import PhoneticModelError, PhoneticOOMError, GGUFLoadError, LLMPatchGenerationError
from .pipeline import PhoneticPipelineError, JSONPatchParseError, PatchApplicationError

__all__ = [
    "PhoneticCorrectionError",
    "PhoneticDatabaseError",
    "TranscriptSaveError",
    "PhoneticModelError",
    "PhoneticOOMError",
    "GGUFLoadError",
    "LLMPatchGenerationError",
    "PhoneticPipelineError",
    "JSONPatchParseError",
    "PatchApplicationError"
]