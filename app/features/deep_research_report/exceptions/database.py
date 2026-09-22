# Path: app/features/deep_research_report/exceptions/database.py

from .base import DeepResearchReportError


class ReportPersistenceError(DeepResearchReportError):
    """Raised when a report, its plan, its sections or its evidence bank cannot be read or written."""
    pass


class ReportNotFoundError(DeepResearchReportError):
    """Raised when a report id does not resolve -- e.g. resuming a report that was deleted."""
    pass
