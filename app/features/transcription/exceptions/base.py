# filename: transcription_exceptions_base_v1.py
# path: app/features/transcription/exceptions/base.py

class TranscriptionError(Exception):
    """
    The Root Exception for the Transcription Feature.
    All transcription-specific exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception