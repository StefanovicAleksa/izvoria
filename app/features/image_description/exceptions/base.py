# Path: app/features/image_description/exceptions/base.py

class ImageDescriptionError(Exception):
    """Root exception of the Image Description feature."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
