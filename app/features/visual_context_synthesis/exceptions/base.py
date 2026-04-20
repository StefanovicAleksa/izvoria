# Canvas: visual_synthesis_exceptions_base_v1
# Path: app/features/visual_context_synthesis/exceptions/base.py

class VisualSynthesisError(Exception):
    """
    The Root Exception for the Visual Context Synthesis feature (Phase 14).
    All custom exceptions in this phase inherit from this to allow the Orchestrator
    to cleanly catch feature-specific failures.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception