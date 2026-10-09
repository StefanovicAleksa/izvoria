# Path: tests/features/report_export/service/test_report_export_api.py

import uuid
from unittest.mock import Mock

import pytest

from app.features.report_export.exceptions import DocumentRenderError
from app.features.report_export.service.report_export_api import ReportExportAPI


class TestReportExportAPI:
    def test_pdf_is_what_the_writer_writes(self):
        writer = Mock()
        writer.write.return_value = b"%PDF-1.7"
        report = Mock(id=uuid.uuid4())

        assert ReportExportAPI(writer).pdf(report) == b"%PDF-1.7"
        writer.write.assert_called_once_with(report)

    def test_a_writer_failure_is_a_typed_export_error_naming_the_report(self):
        writer = Mock()
        writer.write.side_effect = ValueError("layout")
        report = Mock(id=uuid.UUID(int=7))

        with pytest.raises(DocumentRenderError, match=str(uuid.UUID(int=7))):
            ReportExportAPI(writer).pdf(report)

    def test_without_a_writer_it_uses_fpdf(self):
        from app.features.report_export.data.fpdf_report_writer import FpdfReportWriter

        assert isinstance(ReportExportAPI()._pdf_writer, FpdfReportWriter)
