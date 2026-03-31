# Canvas: frame_extraction_phase_v3
# Path: app/orchestration/service/phases/frame_extraction_phase.py

import uuid
from typing import List

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.features.frame_extraction.service.extraction_api import FrameExtractionAPI
from app.features.contextual_retrieval.domain.interfaces import IContextualRepository
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.storage.service.storage_api import StorageAPI

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, VisualCueDetectionJobResult, FrameExtractionJobResult


class FrameExtractionPhase:
    def __init__(
            self,
            job_manager: JobManager,
            extraction_api: FrameExtractionAPI,
            contextual_repo: IContextualRepository,
            storage_api: StorageAPI
    ):
        self._job_manager = job_manager
        self._extraction_api = extraction_api
        self._contextual_repo = contextual_repo
        self._storage = storage_api

    def _process_transcript_frames(self, source_id: uuid.UUID, transcript: ContextualizedTranscript,
                                   video_path: str) -> dict:
        """Helper method executed within the Job context to loop through chunks."""
        extracted_chunks_count = 0

        for chunk in transcript.chunks:
            # We strictly respect the NLP gatekeeper's flag set in Phase 10
            if getattr(chunk, "needs_visual", False):
                self._extraction_api.process_chunk(
                    source_id=source_id,
                    video_path=video_path,
                    chunk=chunk
                )
                extracted_chunks_count += 1

        return {"extracted_chunks_count": extracted_chunks_count}

    def execute(self, cue_detection_results: List[VisualCueDetectionJobResult]) -> PhaseBatchResult:
        """
        Phase 11: Iterates through transcripts evaluated by Phase 10. For any chunk
        flagged with `needs_visual`, delegates to the CPU-bound FrameExtractionAPI.
        """
        results: List[FrameExtractionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0

        for item in cue_detection_results:
            if not item.success:
                continue

            source_id = item.source_id

            # Fetch the original video file path
            source = self._storage.get_source(source_id)
            if not source or not source.file:
                err_msg = f"Cannot extract frames: Video file for Source {source_id} missing."
                results.append(FrameExtractionJobResult(source_id=source_id, success=False, error_message=err_msg))
                errors.append(err_msg)
                failed_count += 1
                continue

            # Fetch the transcript that Phase 10 just updated with boolean flags
            transcript = self._contextual_repo.get_by_source_id(source_id)
            if not transcript:
                err_msg = f"Cannot extract frames: Evaluated Contextual Transcript for Source {source_id} missing."
                results.append(FrameExtractionJobResult(source_id=source_id, success=False, error_message=err_msg))
                errors.append(err_msg)
                failed_count += 1
                continue

            request = JobRequest(
                job_type=JobType.FRAME_EXTRACTION,
                source_id=source_id,
                input_params={"video_source_id": str(source_id)}
            )

            try:
                # Execute API via JobManager to track execution state
                job_result_data = self._job_manager.run_job(
                    request,
                    self._process_transcript_frames,
                    source_id=source_id,
                    transcript=transcript,
                    video_path=source.file.file_path
                )

                extracted_chunks_count = job_result_data.get("extracted_chunks_count", 0)

                results.append(FrameExtractionJobResult(
                    source_id=source_id,
                    success=True,
                    extracted_chunks_count=extracted_chunks_count
                ))
                success_count += 1

            except Exception as e:
                err_msg = str(e)
                results.append(FrameExtractionJobResult(source_id=source_id, success=False, error_message=err_msg))
                errors.append(f"Failed Frame Extraction for Source {source_id}: {err_msg}")
                failed_count += 1

        if failed_count == 0 and success_count > 0:
            status = PhaseStatus.COMPLETED
        elif success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS
        else:
            status = PhaseStatus.COMPLETED

        return PhaseBatchResult(
            phase=PipelinePhase.FRAME_EXTRACTION,
            status=status,
            total_processed=len(cue_detection_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )