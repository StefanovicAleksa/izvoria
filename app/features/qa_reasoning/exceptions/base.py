# Path: app/features/qa_reasoning/exceptions/base.py

class QAReasoningError(Exception):
    """
    The Root Exception for the QA Reasoning (Deep Research Reasoner) feature.
    All other reasoning exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
