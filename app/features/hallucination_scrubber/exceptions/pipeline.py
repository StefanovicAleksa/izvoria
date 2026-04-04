# Canvas: scrubber_exceptions_pipeline_v1
# Path: app/features/hallucination_scrubber/exceptions/pipeline.py

from .base import HallucinationScrubberError

class ScrubberPipelineError(HallucinationScrubberError):
    """Base exception for text parsing and data-flow errors."""
    pass

class TextSegmentationError(ScrubberPipelineError):
    """Raised when the system fails to split a flagged transcript segment into valid sub-clauses."""
    pass

class ScrubbingApplicationError(ScrubberPipelineError):
    """Raised when the system fails to safely strip the hallucinated text while preserving the original timestamps."""
    pass