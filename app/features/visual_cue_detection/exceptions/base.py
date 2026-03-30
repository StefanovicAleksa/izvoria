# Canvas: visual_cue_exceptions_base_v1
# Path: app/features/visual_cue_detection/exceptions/base.py

class VisualCueDetectionError(Exception):
    """The Root Exception for the Visual Cue Detection feature."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception