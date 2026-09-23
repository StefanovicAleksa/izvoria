# Path: app/features/document_conversion/exceptions/base.py

class DocumentConversionError(Exception):
    """Root exception of the Document Conversion feature."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
