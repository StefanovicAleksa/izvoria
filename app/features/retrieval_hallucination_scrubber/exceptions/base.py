# Canvas: base_v1
# Path: app/features/retrieval_hallucination_scrubber/exceptions/base.py

class RetrievalScrubberError(Exception):
    """
    The Root Exception for the Retrieval Hallucination Scrubber feature.
    All custom exceptions in this module inherit from this to allow the Orchestrator
    to catch feature-specific failures cleanly.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception