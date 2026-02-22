# filename: orchestration_service_phases_extract_audio_phase_v1.py
import uuid
from typing import List

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, AudioExtractionJobResult


class ExtractAudioPhase:
    def __init__(self, job_manager: JobManager, audio_extractor: AudioExtractorAPI):
        self._job_manager = job_manager
        self._audio_extractor = audio_extractor

    def execute(self, video_source_ids: List[uuid.UUID]) -> PhaseBatchResult:
        """
        Phase 3: Extracts audio from a batch of ingested Video Source IDs.
        Delegates the heavy FFmpeg work to the AudioExtractorAPI.
        """
        results: List[AudioExtractionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0

        for source_id in video_source_ids:
            request = JobRequest(
                job_type=JobType.AUDIO_EXTRACTION,
                source_id=source_id,  # Link the job history directly to the Source
                input_params={"video_source_id": str(source_id)}
            )

            try:
                # Let JobManager lock and track the FFmpeg execution
                extraction = self._job_manager.run_job(
                    request,
                    self._audio_extractor.extract_audio_from_video,
                    video_source_id=source_id
                )

                result = AudioExtractionJobResult(
                    source_id=source_id,
                    success=True,
                    extracted_file_id=extraction.extracted_file_id
                )
                results.append(result)
                success_count += 1

            except Exception as e:
                err_msg = str(e)
                result = AudioExtractionJobResult(
                    source_id=source_id,
                    success=False,
                    error_message=err_msg
                )
                results.append(result)
                errors.append(f"Failed to extract audio for {source_id}: {err_msg}")
                failed_count += 1

        # Determine overall phase status
        if failed_count == 0 and success_count > 0:
            status = PhaseStatus.COMPLETED
        elif success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS
        else:
            status = PhaseStatus.COMPLETED  # Empty batch

        return PhaseBatchResult(
            phase=PipelinePhase.AUDIO_EXTRACTION,
            status=status,
            total_processed=len(video_source_ids),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )