# File: app/features/vad/exceptions/base.py

class VadError(Exception):
    """
    The Root Exception for the VAD Feature.
    All VAD-specific exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception