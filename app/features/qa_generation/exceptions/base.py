# Canvas: qa_generation_exceptions_base_v1
# Path: app/features/qa_generation/exceptions/base.py

class QAGenerationError(Exception):
    """
    The Root Exception for the QA Generation Feature.
    All other generation exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception