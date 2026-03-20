class AsrCorrectionError(Exception):
    """
    The Root Exception for the ASR Error Correction Feature.
    All exceptions in the 3-pass pipeline inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception