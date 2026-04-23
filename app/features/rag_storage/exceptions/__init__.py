# Canvas: rag_storage_exceptions_init_v1
# Path: app/features/rag_storage/exceptions/__init__.py

from .base import RAGStorageError
from .database import (
    VectorDatabaseError,
    SparseIndexError,
    IdempotencyError
)
from .model import (
    EmbeddingError,
    EmbeddingModelNotLoadedError,
    EmbeddingInferenceError
)

__all__ = [
    "RAGStorageError",
    "VectorDatabaseError",
    "SparseIndexError",
    "IdempotencyError",
    "EmbeddingError",
    "EmbeddingModelNotLoadedError",
    "EmbeddingInferenceError",
]