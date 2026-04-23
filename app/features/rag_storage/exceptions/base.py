# Path: app/features/rag_storage/exceptions/base.py

class RAGStorageError(Exception):
    """Base exception for all RAG Storage related failures."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception