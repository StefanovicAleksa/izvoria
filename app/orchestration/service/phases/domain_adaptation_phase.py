# Canvas: domain_adaptation_phase_v2
# Path: app/orchestration/service/phases/domain_adaptation_phase.py

from typing import List

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.features.domain_adaptation.service.adaptation_api import DomainAdaptationAPI
from app.features.storage.service.storage_api import StorageAPI
from app.features.vad.domain.interfaces import IVadRepository

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, VadJobResult, DomainAdaptationJobResult


class DomainAdaptationPhase:
    def __init__(
            self,
            job_manager: JobManager,
            adaptation_api: DomainAdaptationAPI,
            storage_api: StorageAPI,
            vad_repo: IVadRepository
    ):
        self._job_manager = job_manager
        self._api = adaptation_api
        self._storage = storage_api
        self._vad_repo = vad_repo

    def execute(self, vad_results: List[VadJobResult]) -> PhaseBatchResult:
        """
        Phase 4.5: Examines the audio to detect the industry sub-domain and generates Whisper hotwords.
        """
        results: List[DomainAdaptationJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0

        for item in vad_results:
            if not item.success or not item.vad_map_id:
                continue

            source_id = item.source_id

            # Fetch the original Source to find its linked audio file
            source = self._storage.get_source(source_id)
            if not source or not source.file:
                err_msg = f"Cannot adapt domain: Audio file for Source {source_id} missing."
                results.append(DomainAdaptationJobResult(source_id=source_id, success=False, error_message=err_msg))
                errors.append(err_msg)
                failed_count += 1
                continue

            # Fetch the VAD Map prerequisite
            vad_map = self._vad_repo.get_map_by_source_id(source_id)
            if not vad_map:
                err_msg = f"Cannot adapt domain: VAD Map for Source {source_id} missing."
                results.append(DomainAdaptationJobResult(source_id=source_id, success=False, error_message=err_msg))
                errors.append(err_msg)
                failed_count += 1
                continue

            request = JobRequest(
                job_type=JobType.DOMAIN_ADAPTATION,
                source_id=source_id,
                input_params={"audio_path": source.file.file_path}
            )

            try:
                # Execute API
                # Note: DomainAdaptationAPI manages its own quick GPU loads/unloads internally
                glossary = self._job_manager.run_job(
                    request,
                    self._api.process_domain_adaptation,
                    source_id=source_id,
                    audio_path=source.file.file_path,
                    vad_map=vad_map
                )

                results.append(DomainAdaptationJobResult(
                    source_id=source_id,
                    success=True,
                    glossary_id=glossary.id
                ))
                success_count += 1

            except Exception as e:
                err_msg = str(e)
                results.append(DomainAdaptationJobResult(source_id=source_id, success=False, error_message=err_msg))
                errors.append(f"Failed Domain Adaptation for Source {source_id}: {err_msg}")
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
            phase=PipelinePhase.DOMAIN_ADAPTATION,
            status=status,
            total_processed=len(vad_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )