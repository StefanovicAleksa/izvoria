# Canvas: rag_retrieval_exceptions_storage_v1
# Path: app/features/rag_retrieval/exceptions/storage.py

from .base import RetrievalError

class MissingClipError(RetrievalError):
    """Raised when a search result is found, but the physical .mp4 clip is missing from disk."""
    pass