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