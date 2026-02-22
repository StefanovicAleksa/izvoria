# filename: orchestration_service_phases_ingest_phase_v1.py
from typing import List

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import SourceType

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, IngestJobResult


class IngestPhase:
    def __init__(self, job_manager: JobManager, storage_api: StorageAPI):
        self._job_manager = job_manager
        self._storage = storage_api

    def execute(self, file_paths: List[str], source_type: SourceType) -> PhaseBatchResult:
        """
        Phase 2: Ingests a batch of raw file paths into the storage system.
        Catches item-level failures so the batch can continue processing.
        """
        results: List[IngestJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0

        for path in file_paths:
            request = JobRequest(
                job_type=JobType.FILE_INGEST,
                input_params={"file_path": path, "source_type": source_type.value}
            )

            try:
                # Execute via JobManager to track PENDING -> PROCESSING -> COMPLETED/FAILED
                source = self._job_manager.run_job(
                    request,
                    self._storage.ingest_local_file,
                    file_path=path,
                    source_type=source_type
                )

                result = IngestJobResult(
                    file_path=path,
                    success=True,
                    source_id=source.id
                )
                results.append(result)
                success_count += 1

            except Exception as e:
                # The JobManager already marked this job as FAILED in the DB.
                # We catch it here to prevent the Phase loop from crashing.
                err_msg = str(e)
                result = IngestJobResult(
                    file_path=path,
                    success=False,
                    error_message=err_msg
                )
                results.append(result)
                errors.append(f"Failed to ingest {path}: {err_msg}")
                failed_count += 1

        # Determine overall phase status
        if failed_count == 0 and success_count > 0:
            status = PhaseStatus.COMPLETED
        elif success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS
        else:
            status = PhaseStatus.COMPLETED  # Empty batch handled gracefully

        return PhaseBatchResult(
            phase=PipelinePhase.INGEST,
            status=status,
            total_processed=len(file_paths),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )