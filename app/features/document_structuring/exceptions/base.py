# Path: app/features/document_structuring/exceptions/base.py

class DocumentStructuringError(Exception):
    """Root exception of the Document Structuring feature."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
