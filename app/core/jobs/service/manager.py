# File: app/core/jobs/service/manager.py

import logging
import traceback
import uuid
from typing import Any, Callable, List, Optional

from app.core.jobs.domain.entities import CancellationToken, Job, JobRequest
from app.core.jobs.domain.interfaces import IJobRepository
from app.core.jobs.exceptions.execution import JobCancelled, JobExecutionFailed

logger = logging.getLogger(__name__)

_JSON_SCALARS = (str, int, float, bool, type(None))

# Reported to a progress listener around each unit of work.
PROGRESS_STARTED = "started"
PROGRESS_COMPLETED = "completed"
PROGRESS_FAILED = "failed"
PROGRESS_CANCELLED = "cancelled"


class JobManager:
    def __init__(self, repository: IJobRepository, progress_listener: Optional[Callable] = None):
        """
        `progress_listener(phase: str, state: str, source_id: Optional[UUID], detail: Optional[str])`
        is called around every job. Every Phase wraps its per-file work in run_job, so one listener
        here observes every file of every phase in every pipeline -- which is the only way a
        long-running caller (a GUI) can show batch progress, since the pipelines themselves report
        nothing until they return.

        Defaults to None so the CLI is unaffected.
        """
        self._repo = repository
        self._progress_listener = progress_listener

    def run_job(
            self,
            request: JobRequest,
            target_func: Callable,
            *args,
            cancel_token: Optional[CancellationToken] = None,
            **kwargs
    ) -> Any:
        """
        Executes a feature function within the Job tracking context.
        Records success/failure and bubbles up any underlying exceptions.

        `cancel_token` is checked once, BEFORE the job row is created and before `target_func` runs, so a
        cancelled run never starts work it is about to throw away. It is not checked during the call: the
        unit of work here is whatever the feature does, and interrupting mid-generation is not something
        llama.cpp offers. Granularity is therefore one job -- which for a report is one section.
        """
        if cancel_token is not None and cancel_token.is_cancelled:
            logger.info(f"Skipping {request.job_type.value}: the run was cancelled.")
            self._notify(request, PROGRESS_CANCELLED)
            raise JobCancelled(f"{request.job_type.value} was not started: the run was cancelled.")

        # 1. Announce the job to the database
        job = self._repo.create_job(request)

        # 2. Lock it for execution
        started_job = self._repo.start_job(job.id)
        if not started_job:
            raise JobExecutionFailed(f"Failed to start Job {job.id}: Could not lock for processing.")

        self._notify(request, PROGRESS_STARTED)

        try:
            # 3. Execute the actual feature logic (e.g., Scanner, Ingest, Transcribe)
            result = target_func(*args, **kwargs)

            # 4. Format the output (Database expects a JSON-safe dictionary for JSONB)
            output_data = self._summarize_output(result)

            # 5. Finalize as Success
            self._repo.finish_job(job.id, output_data)
            self._notify(request, PROGRESS_COMPLETED)

            # Return the feature's actual result to the caller (Orchestrator)
            return result

        except Exception as e:
            # 6. Intercept the failure to record it in the Black Box
            error_message = str(e)
            stack_trace = traceback.format_exc()
            self._repo.fail_job(job.id, error_message=error_message, stack_trace=stack_trace)
            self._notify(request, PROGRESS_FAILED, detail=error_message)

            # 7. Bubble up the original exception
            # (so the Orchestrator can catch ScannerError, StorageError, etc.)
            raise e

    def _notify(self, request: JobRequest, state: str, detail: Optional[str] = None) -> None:
        """
        Fires the progress listener. A listener that raises must never turn into a pipeline
        failure -- progress reporting is observation, not part of the work.
        """
        if self._progress_listener is None:
            return
        try:
            self._progress_listener(
                request.job_type.value, state, request.source_id, detail or request.progress_detail
            )
        except Exception as e:
            logger.debug(f"Progress listener raised and was ignored: {e}")

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
