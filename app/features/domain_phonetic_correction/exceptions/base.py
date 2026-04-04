# Canvas: exceptions_base_v1
# Path: app/features/phonetic_correction/exceptions/base.py

class PhoneticCorrectionError(Exception):
    """
    The Root Exception for the Phonetic Correction feature.
    All custom exceptions in Phase 8 inherit from this to allow the Orchestrator
    to catch feature-specific failures cleanly.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception