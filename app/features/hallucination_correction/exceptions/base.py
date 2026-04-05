# Canvas: hallucination_correction_exceptions_base_v1
# Path: app/features/hallucination_correction/exceptions/base.py

class HallucinationCorrectionError(Exception):
    """
    The Root Exception for the Hallucination Correction Feature.
    All correction-specific exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception