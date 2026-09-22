# Path: app/features/deep_research_report/exceptions/model.py

from .base import DeepResearchReportError


class ReportModelError(DeepResearchReportError):
    """Base exception for AI hardware and inference failures in the report model."""
    pass


class ReportOOMError(ReportModelError):
    """Raised when the GPU VRAM limit is breached during report model load or inference."""
    pass


class ReportInferenceError(ReportModelError):
    """Raised when generation fails unexpectedly within the Llama.cpp engine."""
    pass
