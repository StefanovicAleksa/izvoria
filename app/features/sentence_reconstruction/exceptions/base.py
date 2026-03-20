# Canvas: sentence_exceptions_base_v3
# Path: app/features/sentence_reconstruction/exceptions/base.py

class SentenceReconstructionError(Exception):
    """
    Base exception for all errors within the Phase 7 pipeline.
    Allows the Orchestrator and Job Manager to easily catch feature-level faults.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception