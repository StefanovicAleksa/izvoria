# File: integration_tests/core/jobs/test_jobs_integration.py

import pytest
import time
from app.core.database.service.manager import db_manager
from app.core.jobs.data.repositories import JobRepository
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType, JobStatus


@pytest.fixture
def job_repo():
    """Uses the real test database managed by the integration suite."""
    return JobRepository(db_manager)


@pytest.mark.integration
def test_job_lifecycle_end_to_end(job_repo):
    """
    Tests the full state machine of a job in the real database:
    PENDING -> PROCESSING -> COMPLETED
    """
    # 1. CREATE (PENDING)
    # Using DIRECTORY_SCAN so source_id can safely be None without FK violation
    req = JobRequest(
        job_type=JobType.DIRECTORY_SCAN,
        input_params={"target_dir": "/mock/path"}
    )
    job = job_repo.create_job(req)

    assert job.id is not None
    assert job.status == JobStatus.PENDING
    assert job.job_type == JobType.DIRECTORY_SCAN
    assert job.input_params == {"target_dir": "/mock/path"}

    # 2. START (PROCESSING)
    started_job = job_repo.start_job(job.id)

    assert started_job.id == job.id
    assert started_job.status == JobStatus.PROCESSING
    assert started_job.started_at is not None

    # Simulate work taking a tiny bit of time
    time.sleep(0.1)

    # 3. FINISH (COMPLETED)
    finished_job = job_repo.finish_job(job.id, output_data={"files_found": 5})

    assert finished_job.status == JobStatus.COMPLETED
    assert finished_job.completed_at is not None
    assert finished_job.output_result == {"files_found": 5}

    # Verify the dynamically computed property works correctly
    assert finished_job.duration_ms is not None
    assert finished_job.duration_ms > 0


@pytest.mark.integration
def test_job_lifecycle_failure(job_repo):
    """
    Tests the failure path of a job in the real database:
    PENDING -> PROCESSING -> FAILED
    """
    # 1. CREATE
    req = JobRequest(job_type=JobType.DIRECTORY_SCAN)
    job = job_repo.create_job(req)

    # 2. START
    job_repo.start_job(job.id)

    # 3. FAIL
    failed_job = job_repo.fail_job(job.id, error_message="Disk full", stack_trace="Traceback...")

    assert failed_job.status == JobStatus.FAILED
    assert failed_job.error_message == "Disk full"
    assert failed_job.stack_trace == "Traceback..."
    assert failed_job.completed_at is not None