# Canvas: hallucination_correction_exceptions_init_v1
# Path: app/features/hallucination_correction/exceptions/__init__.py

from .base import HallucinationCorrectionError
from .validation import MissingPrerequisiteError, CorruptSegmentDataError
from .database import CorrectionDatabaseError

__all__ = [
    "HallucinationCorrectionError",
    "MissingPrerequisiteError",
    "CorruptSegmentDataError",
    "CorrectionDatabaseError"
]