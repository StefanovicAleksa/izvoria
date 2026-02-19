# File: app/orchestration/service/ingest_flow.py

from typing import List

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.features.scanner.domain.interfaces import IScanner
from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import SourceType

from app.orchestration.domain.entities import ScanJobResult, IngestJobResult
from app.orchestration.exceptions.workflow import ScannerInitializationError


class IngestFlowOrchestrator:
    def __init__(self, job_manager: JobManager, scanner: IScanner, storage_api: StorageAPI):
        self._job_manager = job_manager
        self._scanner = scanner
        self._storage = storage_api

    def run_directory_scan(self, target_directory: str) -> ScanJobResult:
        """
        Phase 1: Executes the directory scan wrapped in a Job context.
        """
        request = JobRequest(
            job_type=JobType.DIRECTORY_SCAN,
            input_params={"target_directory": target_directory}
        )

        # Wrapper function to consume the scanner generator and return serializable data for the Job
        def _scan_wrapper() -> dict:
            file_paths = []
            try:
                for scanned_file in self._scanner.scan(target_directory):
                    file_paths.append(str(scanned_file.path))
                return {"total_found": len(file_paths), "file_paths": file_paths}
            except Exception as e:
                raise ScannerInitializationError(f"Failed to scan directory: {str(e)}", original_exception=e) from e

        # Execute via JobManager to track PENDING -> PROCESSING -> COMPLETED/FAILED
        raw_result = self._job_manager.run_job(request, _scan_wrapper)

        return ScanJobResult(
            target_directory=target_directory,
            total_found=raw_result["total_found"],
            file_paths=raw_result["file_paths"]
        )

    def run_file_ingest(self, file_path: str, source_type: SourceType = SourceType.DOCUMENT) -> IngestJobResult:
        """
        Phase 2: Executes a single file ingest wrapped in a Job context.
        Catches failures so the caller can continue the loop.
        """
        request = JobRequest(
            job_type=JobType.FILE_INGEST,
            input_params={"file_path": file_path, "source_type": source_type.value}
        )

        try:
            # Execute via JobManager
            source = self._job_manager.run_job(
                request,
                self._storage.ingest_local_file,
                file_path=file_path,
                source_type=source_type
            )

            return IngestJobResult(
                file_path=file_path,
                success=True,
                source_id=source.id
            )

        except Exception as e:
            # The JobManager already marked this job as FAILED in the DB.
            # We catch it here to prevent the Orchestrator loop from crashing.
            return IngestJobResult(
                file_path=file_path,
                success=False,
                error_message=str(e)
            )

    def process_full_batch(self, target_directory: str, source_type: SourceType = SourceType.DOCUMENT) -> List[
        IngestJobResult]:
        """
        Convenience method to run Phase 1 and Phase 2 sequentially.
        """
        scan_result = self.run_directory_scan(target_directory)

        ingest_results = []
        for path in scan_result.file_paths:
            result = self.run_file_ingest(path, source_type=source_type)
            ingest_results.append(result)

        return ingest_results