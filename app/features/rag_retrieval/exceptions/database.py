# Canvas: rag_retrieval_exceptions_database_v1
# Path: app/features/rag_retrieval/exceptions/database.py

from .base import RetrievalError

class SearchExecutionError(RetrievalError):
    """Raised when the underlying database fails to execute the hybrid search query."""
    pass

class InvalidSearchSyntaxError(RetrievalError):
    """Raised when the user provides symbols that break Postgres to_tsquery syntax."""
    pass

class IndexLanguageMismatchError(RetrievalError):
    """
    Raised when the index's keyword column was built for a different language than the appliance now serves.
    Its tsvectors were stemmed with the old language's rules, so searching them with the new one's would quietly
    return the wrong chunks, or none.
    """
    pass
