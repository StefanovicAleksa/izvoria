# Canvas: rag_storage_model_exceptions_v1
# Path: app/features/rag_storage/exceptions/model.py

from .base import RAGStorageError

class EmbeddingError(RAGStorageError):
    """Base for embedding model failures."""
    pass

class EmbeddingModelNotLoadedError(EmbeddingError):
    """Raised if trying to embed without calling load() first."""
    pass

class EmbeddingInferenceError(EmbeddingError):
    """Raised when the model fails to process the text batch."""
    pass