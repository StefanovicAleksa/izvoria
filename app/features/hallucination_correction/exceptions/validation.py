# Canvas: hallucination_correction_exceptions_validation_v1
# Path: app/features/hallucination_correction/exceptions/validation.py

from app.features.hallucination_correction.exceptions.base import HallucinationCorrectionError

class MissingPrerequisiteError(HallucinationCorrectionError):
    """
    Raised when the API is asked to process a source, but the required
    Transcript or VAD Map does not exist in the database yet.
    """
    pass

class CorruptSegmentDataError(HallucinationCorrectionError):
    """
    Raised when a transcript segment has mathematically impossible timestamps
    (e.g., start_sec > end_sec), preventing VAD overlap calculations.
    """
    pass