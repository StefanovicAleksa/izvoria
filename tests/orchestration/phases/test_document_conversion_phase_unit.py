# Path: tests/orchestration/phases/test_document_conversion_phase_unit.py

import uuid
from unittest.mock import Mock

from app.core.jobs.domain.enums import JobType
from app.features.document_conversion.domain.entities import ConvertedDocument, PageContent
from app.features.document_conversion.domain.enums import DocumentKind, ExtractionMethod
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase
from app.orchestration.service.phases.document_conversion_phase import DocumentConversionPhase


def _run_job(request, func, **kwargs):
    return func(**kwargs)


class TestDocumentConversionPhase:
    def test_reports_page_counts_per_document(self):
        source_id = uuid.uuid4()
        document = ConvertedDocument(source_id=source_id, kind=DocumentKind.PDF, pages=[
            PageContent(page_no=1, text="text"),
            PageContent(page_no=2, text="ocr", method=ExtractionMethod.OCR, ocr_lines_detected=5, ocr_lines_read=4),
            PageContent(page_no=3, method=ExtractionMethod.NONE),
        ])
        job_manager, api = Mock(), Mock()
        job_manager.run_job.side_effect = _run_job
        api.execute.return_value = document

        result = DocumentConversionPhase(job_manager, api).execute([source_id])

        request = job_manager.run_job.call_args.args[0]
        assert (request.job_type, request.source_id) == (JobType.DOCUMENT_CONVERSION, source_id)
        [item] = result.items
        assert (item.pages_total, item.pages_ocr, item.pages_ocr_incomplete, item.pages_unreadable) == (3, 1, 1, 1)
        assert item.converted_document_id == document.id
        assert (result.phase, result.status) == (PipelinePhase.DOCUMENT_CONVERSION, PhaseStatus.COMPLETED)

    def test_one_failed_document_is_partial_success_and_the_ocr_engine_is_released(self):
        good, bad = uuid.uuid4(), uuid.uuid4()
        job_manager, api = Mock(), Mock()
        job_manager.run_job.side_effect = _run_job
        api.execute.side_effect = lambda source_id: (
            ConvertedDocument(source_id=good, kind=DocumentKind.DOCX) if source_id == good
            else (_ for _ in ()).throw(RuntimeError("password-protected")))

        result = DocumentConversionPhase(job_manager, api).execute([good, bad])

        assert result.status is PhaseStatus.PARTIAL_SUCCESS
        assert [i.success for i in result.items] == [True, False]
        assert "password-protected" in result.errors[0]
        api.release.assert_called_once_with()
