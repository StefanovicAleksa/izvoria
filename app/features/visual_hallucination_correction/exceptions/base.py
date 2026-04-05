# Canvas: visual_hallucination_exceptions_base_v1
# Path: app/features/visual_hallucination_correction/exceptions/base.py

class VisualHallucinationError(Exception):
    """
    The Root Exception for the Visual Hallucination Correction feature (Phase 13).
    All custom exceptions in this module inherit from this to allow the Orchestrator
    to catch feature-specific failures cleanly without crashing the appliance.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception