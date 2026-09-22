# Path: app/features/deep_research_report/exceptions/storage.py

from .base import DeepResearchReportError


class ReportFileError(DeepResearchReportError):
    """Raised when the rendered Markdown cannot be written to disk."""
    pass
