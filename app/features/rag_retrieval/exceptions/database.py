# Canvas: rag_retrieval_exceptions_database_v1
# Path: app/features/rag_retrieval/exceptions/database.py

from .base import RetrievalError

class SearchExecutionError(RetrievalError):
    """Raised when the underlying database fails to execute the hybrid search query."""
    pass

class InvalidSearchSyntaxError(RetrievalError):
    """Raised when the user provides symbols that break Postgres to_tsquery syntax."""
    pass
