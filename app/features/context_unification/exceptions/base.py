# Canvas: context_unification_exceptions_base_v1
# Path: app/features/context_unification/exceptions/base.py

class ContextUnificationError(Exception):
    """
    The Root Exception for the Context Unification feature (Phase 15).
    All custom exceptions in this phase inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception