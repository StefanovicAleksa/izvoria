# File: app/core/jobs/service/manager.py

import traceback
import uuid
from typing import Any, Callable, List

from app.core.jobs.domain.entities import Job, JobRequest
from app.core.jobs.domain.interfaces import IJobRepository
from app.core.jobs.exceptions.execution import JobExecutionFailed

_JSON_SCALARS = (str, int, float, bool, type(None))


class JobManager:
    def __init__(self, repository: IJobRepository):
        self._repo = repository

    def run_job(self, request: JobRequest, target_func: Callable, *args, **kwargs) -> Any:
        """
        Executes a feature function within the Job tracking context.
        Records success/failure and bubbles up any underlying exceptions.
        """
        # 1. Announce the job to the database
        job = self._repo.create_job(request)

        # 2. Lock it for execution
        started_job = self._repo.start_job(job.id)
        if not started_job:
            raise JobExecutionFailed(f"Failed to start Job {job.id}: Could not lock for processing.")

        try:
            # 3. Execute the actual feature logic (e.g., Scanner, Ingest, Transcribe)
            result = target_func(*args, **kwargs)

            # 4. Format the output (Database expects a JSON-safe dictionary for JSONB)
            output_data = self._summarize_output(result)

            # 5. Finalize as Success
            self._repo.finish_job(job.id, output_data)

            # Return the feature's actual result to the caller (Orchestrator)
            return result

        except Exception as e:
            # 6. Intercept the failure to record it in the Black Box
            error_message = str(e)
            stack_trace = traceback.format_exc()
            self._repo.fail_job(job.id, error_message=error_message, stack_trace=stack_trace)

            # 7. Bubble up the original exception
            # (so the Orchestrator can catch ScannerError, StorageError, etc.)
            raise e

    @staticmethod
    def _summarize_output(result: Any) -> dict:
        """
        Builds a compact, JSON-safe record of a job's result.
        Dict results keep their scalar entries (keys starting with "_" carry transient in-memory
        payloads and are dropped); anything else is reduced to its type and id, so whole
        transcripts are never serialized into the jobs table.
        """
        if isinstance(result, dict):
            return {
                str(key): value for key, value in result.items()
                if not str(key).startswith("_") and isinstance(value, _JSON_SCALARS)
            }

        summary = {"result_type": type(result).__name__}
        result_id = getattr(result, "id", None)
        if result_id is not None:
            summary["result_id"] = str(result_id)
        return summary

    def cancel_job(self, job_id: uuid.UUID) -> Job:
        """Manually aborts a job."""
        return self._repo.cancel_job(job_id)

    def get_source_history(self, source_id: uuid.UUID) -> List[Job]:
        """Retrieves the complete job timeline for a specific source."""
        return self._repo.get_job_history(source_id)
