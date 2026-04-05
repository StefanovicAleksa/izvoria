# Canvas: hallucination_correction_phase_v1
# Path: app/orchestration/service/phases/hallucination_correction_phase.py

from typing import List

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.features.hallucination_correction.service.correction_api import CorrectionAPI

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, TranscribeJobResult, HallucinationCorrectionJobResult


class HallucinationCorrectionPhase:
    def __init__(self, job_manager: JobManager, correction_api: CorrectionAPI):
        self._job_manager = job_manager
        self._correction_api = correction_api

    def execute(self, transcribe_results: List[TranscribeJobResult]) -> PhaseBatchResult:
        """
        Phase 5.5: Receives fresh transcripts and surgically amputates known
        hallucinations and dead-air text generation using Zero-VRAM math.
        """
        results: List[HallucinationCorrectionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0

        for item in transcribe_results:
            if not item.success:
                continue

            source_id = item.source_id

            request = JobRequest(
                job_type=JobType.HALLUCINATION_CORRECTION,
                source_id=source_id,
                input_params={"video_source_id": str(source_id)}
            )

            try:
                # The API returns a tuple of (CleanedTranscript, Telemetry)
                cleaned_transcript, telemetry = self._job_manager.run_job(
                    request,
                    self._correction_api.process_transcript,
                    source_id=source_id
                )

                results.append(HallucinationCorrectionJobResult(
                    source_id=source_id,
                    success=True,
                    transcript_id=cleaned_transcript.id
                ))
                success_count += 1

            except Exception as e:
                err_msg = str(e)
                results.append(HallucinationCorrectionJobResult(
                    source_id=source_id,
                    success=False,
                    error_message=err_msg
                ))
                errors.append(f"Failed Hallucination Correction for Source {source_id}: {err_msg}")
                failed_count += 1

        # Determine overall phase status
        if failed_count == 0 and success_count > 0:
            status = PhaseStatus.COMPLETED
        elif success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS
        else:
            status = PhaseStatus.COMPLETED

        return PhaseBatchResult(
            phase=PipelinePhase.HALLUCINATION_CORRECTION,
            status=status,
            total_processed=len(transcribe_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )