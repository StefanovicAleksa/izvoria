# Path: app/features/source_profile/exceptions/base.py

class SourceProfileError(Exception):
    """
    The Root Exception for the Source Profile feature.
    All other source profile exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
