# Canvas: rag_retrieval_exceptions_model_v1
# Path: app/features/rag_retrieval/exceptions/model.py

from .base import RetrievalError

class PromptEmbeddingError(RetrievalError):
    """Raised when the embedding model fails to vectorize the prompt."""
    pass

class ContextLengthExceededError(PromptEmbeddingError):
    """Raised when the user's prompt exceeds the maximum token limit of the embedding model."""
    pass