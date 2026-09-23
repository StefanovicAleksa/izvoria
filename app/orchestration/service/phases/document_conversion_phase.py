# Path: app/orchestration/service/phases/document_conversion_phase.py

import logging
import uuid
from typing import List

from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType
from app.core.jobs.service.manager import JobManager
from app.features.document_conversion.service.conversion_api import DocumentConversionAPI
from app.orchestration.domain.entities import DocumentConversionJobResult, PhaseBatchResult
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase

logger = logging.getLogger(__name__)


class DocumentConversionPhase:
    """PDF/DOCX -> page-tagged Markdown, with OCR for scanned pages. CPU only: no model preload, no VRAM."""

    def __init__(self, job_manager: JobManager, conversion_api: DocumentConversionAPI):
        self._job_manager = job_manager
        self._conversion_api = conversion_api

    def execute(self, source_ids: List[uuid.UUID]) -> PhaseBatchResult:
        results: List[DocumentConversionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0

        try:
            for source_id in source_ids:
                request = JobRequest(
                    job_type=JobType.DOCUMENT_CONVERSION,
                    source_id=source_id,
                    input_params={"document_source_id": str(source_id)}
                )
                try:
                    document = self._job_manager.run_job(request, self._conversion_api.execute, source_id=source_id)
                    results.append(DocumentConversionJobResult(
                        source_id=source_id,
                        success=True,
                        converted_document_id=document.id,
                        pages_total=len(document.pages),
                        pages_ocr=document.pages_ocr,
                        pages_ocr_incomplete=document.pages_ocr_incomplete,
                        pages_unreadable=document.pages_unreadable,
                    ))
                    success_count += 1
                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Document conversion failed for Source {source_id}: {err_msg}")
                    results.append(DocumentConversionJobResult(source_id=source_id, success=False,
                                                               error_message=err_msg))
                    errors.append(f"Failed Document Conversion for Source {source_id}: {err_msg}")
                    failed_count += 1
        finally:
            self._conversion_api.release()

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.DOCUMENT_CONVERSION,
            status=status,
            total_processed=len(source_ids),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )
