# File: tests/core/jobs/test_repository.py

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from app.core.database import exceptions as db_exc
from app.core.jobs.data.repositories import JobRepository
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType, JobStatus
from app.core.jobs.exceptions.system import JobSystemError
from app.core.jobs.exceptions.execution import JobDataError


class TestJobRepository:
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock()
        mock_session_ctx = MagicMock()
        db_manager.session.return_value = mock_session_ctx
        session = MagicMock()
        mock_session_ctx.__enter__.return_value = session
        return db_manager

    @pytest.fixture
    def repository(self, mock_db_manager):
        return JobRepository(mock_db_manager)

    def _create_clean_mock_model(self):
        """Helper to create a mock model with None for unused fields so Pydantic doesn't crash."""
        mock_model = MagicMock()
        mock_model.id = uuid.uuid4()
        mock_model.job_type = "directory_scan"
        mock_model.status = "processing"
        mock_model.created_at = datetime.now(timezone.utc)
        mock_model.input_params = {}
        mock_model.output_params = {}
        # Explicitly set unused fields to None to bypass MagicMock validation errors
        mock_model.source_id = None
        mock_model.worker_id = None
        mock_model.error_message = None
        mock_model.stack_trace = None
        mock_model.started_at = None
        mock_model.completed_at = None
        return mock_model

    # --- CREATE JOB ---
    def test_create_job_success(self, repository, mock_db_manager):
        req = JobRequest(job_type=JobType.DIRECTORY_SCAN, input_params={"path": "/tmp"})
        result = repository.create_job(req)

        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.assert_called_once()
        assert result.job_type == JobType.DIRECTORY_SCAN
        assert result.status == JobStatus.PENDING
        assert result.input_params == {"path": "/tmp"}

    def test_create_job_db_error(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.side_effect = db_exc.SystemError("DB Offline")

        req = JobRequest(job_type=JobType.DIRECTORY_SCAN)
        with pytest.raises(JobSystemError):
            repository.create_job(req)

    # --- START JOB ---
    def test_start_job_success(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        mock_model = self._create_clean_mock_model()

        session.execute.return_value.scalar_one_or_none.return_value = mock_model

        result = repository.start_job(mock_model.id)
        assert result is not None
        assert result.status == JobStatus.PROCESSING

    def test_start_job_not_found(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.execute.return_value.scalar_one_or_none.return_value = None

        result = repository.start_job(uuid.uuid4())
        assert result is None

    # --- FINISH JOB ---
    def test_finish_job_success(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        mock_model = self._create_clean_mock_model()

        session.get.return_value = mock_model
        result = repository.finish_job(mock_model.id, {"found": 10})

        assert mock_model.status == "completed"
        assert mock_model.output_params == {"found": 10}
        assert result.output_result == {"found": 10}
        assert mock_model.completed_at is not None
        assert result.status == JobStatus.COMPLETED

    def test_finish_job_not_found(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.get.return_value = None

        with pytest.raises(JobDataError, match="not found"):
            repository.finish_job(uuid.uuid4(), {})

    # --- FAIL JOB ---
    def test_fail_job_success(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        mock_model = self._create_clean_mock_model()

        session.get.return_value = mock_model
        result = repository.fail_job(mock_model.id, "Crashed!", "Stack trace...")

        assert mock_model.status == "failed"
        assert mock_model.error_message == "Crashed!"
        assert mock_model.stack_trace == "Stack trace..."
        assert result.status == JobStatus.FAILED

    # --- GET HISTORY ---
    def test_get_job_history(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        mock_model = self._create_clean_mock_model()
        mock_model.job_type = "file_ingest"  # <-- FIXED: Valid Enum String
        mock_model.status = "completed"

        session.execute.return_value.scalars.return_value.all.return_value = [mock_model]
        results = repository.get_job_history(uuid.uuid4())

        assert len(results) == 1
        assert results[0].status == JobStatus.COMPLETED