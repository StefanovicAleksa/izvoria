class AudioExtractorError(Exception):
    """
    The Root Exception for the Audio Extractor feature.
    All other extractor exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception

class AudioExtractionDatabaseError(AudioExtractorError):
    """
    Raised when the database fails to save or retrieve an extraction record.
    """
    pass