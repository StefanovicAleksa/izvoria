# Path: app/features/document_structuring/exceptions/structuring.py

from .base import DocumentStructuringError


class EmptyDocumentError(DocumentStructuringError):
    """No page of the document has any text to index: every page was unreadable or blank."""
    pass


class ChunkingError(DocumentStructuringError):
    """Splitting the document's blocks into chunks failed."""
    pass
