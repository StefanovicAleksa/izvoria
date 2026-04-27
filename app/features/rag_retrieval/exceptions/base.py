# Canvas: rag_retrieval_exceptions_base_v1
# Path: app/features/rag_retrieval/exceptions/base.py

class RetrievalError(Exception):
    """Base exception for all RAG Retrieval related failures."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception