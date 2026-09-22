# Path: app/features/deep_research_report/exceptions/base.py


class DeepResearchReportError(Exception):
    """
    The Root Exception for the Deep Research Report feature.
    All other report exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
