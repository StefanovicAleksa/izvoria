# Canvas: visual_analysis_exceptions_base_v1
# Path: app/features/visual_analysis/exceptions/base.py

class VisualAnalysisError(Exception):
    """
    The Root Exception for the Visual Analysis feature (Phase 10).
    All exceptions in this module inherit from this to allow the orchestrator
    to catch feature-specific failures cleanly.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception