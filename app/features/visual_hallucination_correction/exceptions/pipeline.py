# Canvas: visual_hallucination_exceptions_pipeline_v1
# Path: app/features/visual_hallucination_correction/exceptions/pipeline.py

from .base import VisualHallucinationError

class HallucinationPipelineError(VisualHallucinationError):
    """Base exception for parsing, data-flow, and schema validation errors."""
    pass

class JSONAuditParseError(HallucinationPipelineError):
    """
    Raised when the SLM hallucinates invalid JSON instead of adhering to the strict
    boolean audit schema (is_match, discard_reason).
    """
    pass