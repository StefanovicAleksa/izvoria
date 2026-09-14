# Path: app/features/rag_reranking/exceptions/base.py

class RerankingError(Exception):
    """Base exception for all RAG reranking failures."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
