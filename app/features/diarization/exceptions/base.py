# Canvas: diarization_exceptions_base_v1
# Path: app/features/diarization/exceptions/base.py

class DiarizationError(Exception):
    """
    The Root Exception for the Diarization Feature.
    All diarization-specific exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception