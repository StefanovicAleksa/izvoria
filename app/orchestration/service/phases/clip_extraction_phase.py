# Canvas: clip_extraction_phase_v1
# Path: app/orchestration/service/phases/clip_extraction_phase.py

import logging
from typing import List

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.features.clip_extractor.service.extractor_api import ClipExtractorAPI
from app.features.context_unification.domain.interfaces import IContextUnificationRepository

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, ContextUnificationJobResult, ClipExtractionJobResult

logger = logging.getLogger(__name__)


class ClipExtractionPhase:
    """
    Phase 15.5: Takes the finalized Unified Transcripts and extracts physical .mp4 clips
    for every chunk using FFmpeg. CPU/IO Bound.
    """
    def __init__(
            self,
            job_manager: JobManager,
            extractor_api: ClipExtractorAPI,
            unified_repo: IContextUnificationRepository
    ):
        self._job_manager = job_manager
        self._extractor_api = extractor_api
        self._unified_repo = unified_repo

    def execute(self, unification_results: List[ContextUnificationJobResult]) -> PhaseBatchResult:
        logger.info(f"Executing Clip Extraction Phase for {len(unification_results)} transcripts...")
        results: List[ClipExtractionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0

        for item in unification_results:
            if not item.success:
                continue

            source_id = item.source_id

            # Fetch the Unified Transcript prepared in Phase 15
            transcript = self._unified_repo.get_by_source_id(source_id)
            if not transcript:
                err_msg = f"Cannot extract clips: Unified Transcript for Source {source_id} missing."
                logger.error(err_msg)
                results.append(ClipExtractionJobResult(source_id=source_id, success=False, error_message=err_msg))
                errors.append(err_msg)
                failed_count += 1
                continue

            request = JobRequest(
                job_type=JobType.CLIP_EXTRACTION,
                source_id=source_id,
                input_params={"video_source_id": str(source_id)}
            )

            try:
                logger.info(f"Processing Clip Extraction for Source: {source_id}")

                # Execute API
                extracted_clips = self._job_manager.run_job(
                    request,
                    self._extractor_api.extract_clips_from_transcript,
                    transcript=transcript
                )

                results.append(ClipExtractionJobResult(
                    source_id=source_id,
                    success=True,
                    extracted_clips_count=len(extracted_clips)
                ))
                success_count += 1

            except Exception as e:
                err_msg = str(e)
                logger.error(f"Job failed for Source {source_id}: {err_msg}")
                results.append(ClipExtractionJobResult(source_id=source_id, success=False, error_message=err_msg))
                errors.append(f"Failed Clip Extraction for Source {source_id}: {err_msg}")
                failed_count += 1

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.CLIP_EXTRACTION,
            status=status,
            total_processed=len(unification_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )