# Canvas: visual_synthesis_exceptions_process_v1
# Path: app/features/visual_context_synthesis/exceptions/process.py

from .base import VisualSynthesisError

class VisualSynthesisPipelineError(VisualSynthesisError):
    """Base exception for data-flow, sorting, and extraction errors."""
    pass

class FuzzyGroupingError(VisualSynthesisPipelineError):
    """
    Raised when the Jaccard Similarity deduplication algorithm encounters
    malformed data, unexpected types, or fails to logically group the time-spans.
    """
    pass