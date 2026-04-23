# Canvas: rag_storage_db_exceptions_v1
# Path: app/features/rag_storage/exceptions/database.py

from .base import RAGStorageError

class VectorDatabaseError(RAGStorageError):
    """Raised when pgvector operations fail."""
    pass

class SparseIndexError(RAGStorageError):
    """Raised when BM25 file operations or indexing fail."""
    pass

class IdempotencyError(RAGStorageError):
    """Raised when failing to clear existing data before re-ingestion."""
    pass