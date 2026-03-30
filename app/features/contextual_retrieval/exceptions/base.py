# Canvas: contextual_exceptions_base_v1
# Path: app/features/contextual_retrieval/exceptions/base.py

class ContextualRetrievalError(Exception):
    """
    The Root Exception for the Contextual Retrieval feature.
    All exceptions in Phase 9 inherit from this to allow the Orchestrator
    to catch feature-specific failures cleanly.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception