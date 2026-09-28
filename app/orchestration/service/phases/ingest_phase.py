# Canvas: ingest_phase_v2
# Path: app/orchestration/service/phases/ingest_phase.py

from typing import List
from pathlib import Path

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.features.scanner.utils.filters import VIDEO_EXTENSIONS
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
        Phase 2: Ingests a batch of raw file paths.
        Strictly filters for Video files if source_type is VIDEO.
        """
        results: List[IngestJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0

        for path in file_paths:
            # --- NEW FILTER LOGIC ---
            file_path_obj = Path(path)
            if source_type == SourceType.VIDEO:
                if file_path_obj.suffix.lower() not in VIDEO_EXTENSIONS:
                    # Silently ignore non-video files like .pdf or .txt
                    continue

            request = JobRequest(
                job_type=JobType.FILE_INGEST,
                input_params={"file_path": path, "source_type": source_type.value}
            )

            try:
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
                err_msg = str(e)
                result = IngestJobResult(
                    file_path=path,
                    success=False,
                    error_message=err_msg
                )
                results.append(result)
                errors.append(f"Failed to ingest {path}: {err_msg}")
                failed_count += 1

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.INGEST,
            status=status,
            total_processed=len(results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )