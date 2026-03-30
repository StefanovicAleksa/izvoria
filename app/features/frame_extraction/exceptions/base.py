# Canvas: frame_extraction_exceptions_base_v1
# Path: app/features/frame_extraction/exceptions/base.py

class FrameExtractionError(Exception):
    """The Root Exception for the CPU Frame Extraction feature."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception