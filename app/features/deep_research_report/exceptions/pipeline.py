# Path: app/features/deep_research_report/exceptions/pipeline.py

from .base import DeepResearchReportError


class JSONPlanParseError(DeepResearchReportError):
    """
    Raised when the planner returns something that is not a usable plan -- invalid JSON, or a plan with no
    sections. Fails closed: there is no safe default outline, and proposing an invented one would put the
    user's approval on a plan the corpus was never consulted about.
    """
    pass


class OutlineValidationError(DeepResearchReportError):
    """
    Raised when an approved outline coming back from the user is not internally consistent -- duplicate
    section keys, a sub-question pointing at a section that no longer exists, or more sections than
    REPORT_MAX_SECTIONS allows. Rejected before the long run starts rather than halfway through it.
    """
    pass


class ReportCancelledError(DeepResearchReportError):
    """
    Raised when a run is stopped by the user. Not a failure: the sections written so far are kept and the
    report is resumable, so callers catch this and return the partial report rather than propagating.
    """
    pass
