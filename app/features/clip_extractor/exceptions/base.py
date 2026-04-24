# Canvas: clip_extractor_exceptions_base_v1
# Path: app/features/clip_extractor/exceptions/base.py

class ClipExtractorError(Exception):
    """
    The Root Exception for the Clip Extractor feature.
    All other clip extractor exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception