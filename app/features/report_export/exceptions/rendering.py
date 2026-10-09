# Path: app/features/report_export/exceptions/rendering.py

from app.features.report_export.exceptions.base import ReportExportError


class DocumentRenderError(ReportExportError):
    """The document could not be laid out or written (a layout bug, a broken font, a malformed report)."""
    pass


class FontMissingError(ReportExportError):
    """A bundled font file is missing from data/fonts; the export cannot look right without it, so it refuses."""
    pass


class MathRenderError(ReportExportError):
    """One formula could not be typeset. Caught inside the writer, which prints the LaTeX instead."""
    pass
