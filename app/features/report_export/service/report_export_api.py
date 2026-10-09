# Path: app/features/report_export/service/report_export_api.py

from typing import Optional

from app.features.deep_research_report.domain.entities import Report
from app.features.report_export.domain.interfaces import IReportDocumentWriter
from app.features.report_export.utils.error_mapper import map_export_error


class ReportExportAPI:
    """
    Exports a research report as a document file. No model and no database: it lays out the report it is given,
    so the caller decides whose report it is and reads it first.

    Thread-safe: every export builds its own document, and the writer is pure Python (fpdf2), so a server can run
    exports on worker threads.
    """

    def __init__(self, pdf_writer: Optional[IReportDocumentWriter] = None):
        if pdf_writer is None:
            from app.features.report_export.data.fpdf_report_writer import FpdfReportWriter
            pdf_writer = FpdfReportWriter()
        self._pdf_writer = pdf_writer

    def pdf(self, report: Report) -> bytes:
        """The report as a PDF. Raises a ReportExportError subclass if it cannot be made."""
        try:
            return self._pdf_writer.write(report)
        except Exception as e:
            raise map_export_error(e, context=f"Exporting report {report.id} as PDF") from e
