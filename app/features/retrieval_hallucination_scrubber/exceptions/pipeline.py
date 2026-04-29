# Canvas: pipeline_v1
# Path: app/features/retrieval_hallucination_scrubber/exceptions/pipeline.py

from .base import RetrievalScrubberError

class JSONScrubParseError(RetrievalScrubberError):
    """
    Raised when the SLM hallucinates invalid JSON instead of adhering to the strict
    boolean audit schema (is_relevant, discard_reason).
    """
    pass