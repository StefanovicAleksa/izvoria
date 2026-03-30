# Canvas: visual_cue_exceptions_detection_v1
# Path: app/features/visual_cue_detection/exceptions/detection.py

from .base import VisualCueDetectionError

class LLMCueGenerationError(VisualCueDetectionError):
    """Raised when the local SLM crashes or runs out of VRAM during cue detection."""
    pass

class CueParseError(VisualCueDetectionError):
    """Raised when the SLM fails to output valid JSON matching our schema."""
    pass