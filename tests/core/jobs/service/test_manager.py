# File: tests/core/jobs/test_manager.py

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest, Job
from app.core.jobs.domain.enums import JobType, JobStatus
from app.core.jobs.exceptions.execution import JobExecutionFailed


class TestJobManager:
    @pytest.fixture
    def mock_repo(self):
        return Mock()

    @pytest.fixture
    def manager(self, mock_repo):
        return JobManager(mock_repo)

    def test_run_job_happy_path(self, manager, mock_repo):
        """Verifies successful job lifecycle and value return."""
        # 1. Setup Data
        req = JobRequest(job_type=JobType.DIRECTORY_SCAN)

        # We need a mocked Job that gets returned by create and start
        mock_job = Job(
            id=uuid.uuid4(),
            job_type=JobType.DIRECTORY_SCAN,
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc)  # FIXED: Now using actual datetime
        )
        mock_repo.create_job.return_value = mock_job
        mock_repo.start_job.return_value = mock_job

        # 2. Setup Target Function
        def successful_feature_task(multiplier):
            return {"score": 10 * multiplier}

        # 3. Execute
        result = manager.run_job(req, successful_feature_task, multiplier=5)

        # 4. Assertions
        assert result == {"score": 50}
        mock_repo.create_job.assert_called_once_with(req)
        mock_repo.start_job.assert_called_once_with(mock_job.id)
        mock_repo.finish_job.assert_called_once_with(mock_job.id, {"score": 50})
        mock_repo.fail_job.assert_not_called()

    def test_run_job_execution_failure(self, manager, mock_repo):
        """Verifies that if the target function crashes, the job is failed and error bubbles up."""
        # 1. Setup Data
        req = JobRequest(job_type=JobType.DIRECTORY_SCAN)
        mock_job = Job(
            id=uuid.uuid4(),
            job_type=JobType.DIRECTORY_SCAN,
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc)  # FIXED
        )
        mock_repo.create_job.return_value = mock_job
        mock_repo.start_job.return_value = mock_job

        # 2. Setup Target Function (Crash)
        def failing_task():
            raise ValueError("Unexpected feature crash!")

        # 3. Execute & Assert Exception Bubbles Up
        with pytest.raises(ValueError, match="Unexpected feature crash!"):
            manager.run_job(req, failing_task)

        # 4. Assertions on Repo
        mock_repo.create_job.assert_called_once()
        mock_repo.start_job.assert_called_once()
        mock_repo.finish_job.assert_not_called()

        # Verify fail_job caught the details
        mock_repo.fail_job.assert_called_once()
        call_args = mock_repo.fail_job.call_args[1]
        assert call_args["error_message"] == "Unexpected feature crash!"
        assert "Traceback" in call_args["stack_trace"]

    def test_run_job_lock_failure(self, manager, mock_repo):
        """Verifies that if start_job returns None, JobExecutionFailed is raised."""
        req = JobRequest(job_type=JobType.DIRECTORY_SCAN)
        mock_job = Job(
            id=uuid.uuid4(),
            job_type=JobType.DIRECTORY_SCAN,
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc)  # FIXED
        )

        mock_repo.create_job.return_value = mock_job
        # Simulate locking failure (e.g., job already grabbed or disappeared)
        mock_repo.start_job.return_value = None

        def dummy_task():
            pass

        with pytest.raises(JobExecutionFailed, match="Could not lock for processing"):
            manager.run_job(req, dummy_task)

        mock_repo.finish_job.assert_not_called()
        mock_repo.fail_job.assert_not_called()

    def _started_job(self, mock_repo):
        job = Job(
            id=uuid.uuid4(),
            job_type=JobType.DIRECTORY_SCAN,
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc)
        )
        mock_repo.create_job.return_value = job
        mock_repo.start_job.return_value = job
        return job

    def test_run_job_drops_transient_and_non_scalar_dict_entries(self, manager, mock_repo):
        job = self._started_job(mock_repo)
        payload = {"surviving_chunks_count": 2, "_transient_chunks": [object()], "nested": {"a": 1}}

        result = manager.run_job(JobRequest(job_type=JobType.DIRECTORY_SCAN), lambda: payload)

        assert result is payload
        mock_repo.finish_job.assert_called_once_with(job.id, {"surviving_chunks_count": 2})

    def test_run_job_summarizes_domain_objects_by_type_and_id(self, manager, mock_repo):
        job = self._started_job(mock_repo)

        class FakeTranscript:
            def __init__(self):
                self.id = uuid.uuid4()
                self.segments = ["huge"] * 1000

        transcript = FakeTranscript()
        manager.run_job(JobRequest(job_type=JobType.DIRECTORY_SCAN), lambda: transcript)

        mock_repo.finish_job.assert_called_once_with(
            job.id, {"result_type": "FakeTranscript", "result_id": str(transcript.id)}
        )

    def test_run_job_summarizes_results_without_id(self, manager, mock_repo):
        job = self._started_job(mock_repo)

        manager.run_job(JobRequest(job_type=JobType.DIRECTORY_SCAN), lambda: ("cleaned", "telemetry"))

        mock_repo.finish_job.assert_called_once_with(job.id, {"result_type": "tuple"})


class TestJobManagerProgressListener:
    """
    The progress listener is how a long-running caller (the desktop app) observes a batch run.
    Every Phase routes its per-file work through run_job, so this one hook covers every file of
    every phase -- which is the only reason batch progress is visible at all.
    """

    @pytest.fixture
    def mock_repo(self):
        repo = Mock()
        job = Job(
            id=uuid.uuid4(),
            job_type=JobType.TRANSCRIBE,
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        repo.create_job.return_value = job
        repo.start_job.return_value = job
        return repo

    @pytest.fixture
    def source_id(self):
        return uuid.uuid4()

    def test_listener_is_optional(self, mock_repo):
        """The CLI constructs JobManager with no listener and must be unaffected."""
        manager = JobManager(mock_repo)
        req = JobRequest(job_type=JobType.TRANSCRIBE)

        assert manager.run_job(req, lambda: "done") == "done"

    def test_reports_started_then_completed_with_the_phase_and_source(self, mock_repo, source_id):
        events = []
        manager = JobManager(mock_repo, progress_listener=lambda *args: events.append(args))
        req = JobRequest(job_type=JobType.TRANSCRIBE, source_id=source_id)

        manager.run_job(req, lambda: "transcript")

        assert events == [
            ("transcribe", "started", source_id, None),
            ("transcribe", "completed", source_id, None),
        ]

    def test_reports_failed_with_the_error_message_and_still_raises(self, mock_repo, source_id):
        events = []
        manager = JobManager(mock_repo, progress_listener=lambda *args: events.append(args))
        req = JobRequest(job_type=JobType.TRANSCRIBE, source_id=source_id)

        def exploding_task():
            raise ValueError("CUDA out of memory")

        with pytest.raises(ValueError):
            manager.run_job(req, exploding_task)

        assert events == [
            ("transcribe", "started", source_id, None),
            ("transcribe", "failed", source_id, "CUDA out of memory"),
        ]

    def test_a_raising_listener_cannot_fail_the_job(self, mock_repo, source_id):
        """
        Progress reporting is observation. If a listener throws -- a disconnected Qt signal, say --
        the pipeline must carry on rather than losing an hour of transcription to a UI bug.
        """
        def bad_listener(*_args):
            raise RuntimeError("the UI went away")

        manager = JobManager(mock_repo, progress_listener=bad_listener)
        req = JobRequest(job_type=JobType.TRANSCRIBE, source_id=source_id)

        assert manager.run_job(req, lambda: "transcript") == "transcript"
        mock_repo.finish_job.assert_called_once()

    def test_no_progress_is_reported_when_the_job_cannot_be_locked(self, mock_repo):
        """A job that never started must not report that it did."""
        events = []
        mock_repo.start_job.return_value = None
        manager = JobManager(mock_repo, progress_listener=lambda *args: events.append(args))
        req = JobRequest(job_type=JobType.TRANSCRIBE)

        with pytest.raises(JobExecutionFailed):
            manager.run_job(req, lambda: "never runs")

        assert events == []
