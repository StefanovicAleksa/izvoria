# filename: orchestration_service_phases_scan_phase_v2.py
from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.features.scanner.domain.interfaces import IScanner
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult
from app.orchestration.exceptions.phase import ScannerInitializationError


class ScanPhase:
    def __init__(self, job_manager: JobManager, scanner: IScanner):
        self._job_manager = job_manager
        self._scanner = scanner

    def execute(self, target_directory: str) -> PhaseBatchResult:
        """
        Phase 1: Scans the target directory for supported files.
        Executes as a single systemic Job.
        """
        request = JobRequest(
            job_type=JobType.DIRECTORY_SCAN,
            input_params={"target_directory": target_directory}
        )

        def _scan_wrapper() -> dict:
            scanned_paths = []  # Renamed to avoid shadowing outer scope
            try:
                # Delegate to the pure domain scanner
                for scanned_file in self._scanner.scan(target_directory):
                    scanned_paths.append(str(scanned_file.path))

                return {"total_found": len(scanned_paths), "file_paths": scanned_paths}
            except Exception as scan_err:  # Renamed exception variable
                raise ScannerInitializationError(
                    f"Failed to scan directory: {str(scan_err)}",
                    original_exception=scan_err
                ) from scan_err

        try:
            # Execute the systemic job
            raw_result = self._job_manager.run_job(request, _scan_wrapper)

            extracted_file_paths = raw_result.get("file_paths", [])
            total_found = raw_result.get("total_found", 0)

            # Map to the new Batch Result format
            status = PhaseStatus.COMPLETED if total_found > 0 else PhaseStatus.PARTIAL_SUCCESS

            return PhaseBatchResult(
                phase=PipelinePhase.SCAN,
                status=status,
                total_processed=1,  # 1 Systemic Job
                successful_count=total_found,
                failed_count=0,
                items=extracted_file_paths,  # The Pipeline Director will iterate over these strings
                errors=[]
            )

        except ScannerInitializationError as init_err:  # Renamed exception variable
            # Bubble up critical initialization failures directly
            raise init_err
        except Exception as sys_err:  # Renamed exception variable
            # Catch job manager system crashes safely
            return PhaseBatchResult(
                phase=PipelinePhase.SCAN,
                status=PhaseStatus.FAILED,
                total_processed=0,
                successful_count=0,
                failed_count=1,
                items=[],
                errors=[str(sys_err)]
            )