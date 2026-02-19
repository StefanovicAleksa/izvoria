# File: app/core/jobs/data/repositories.py

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import select, update

from app.core.database.domain.interfaces import IDatabaseManager
from app.core.database import exceptions as db_exc
from app.core.jobs.domain.interfaces import IJobRepository
from app.core.jobs.domain.entities import Job, JobRequest
from app.core.jobs.domain.enums import JobStatus
from app.core.jobs.data.models import JobModel
from app.core.jobs.exceptions.execution import JobDataError
from app.core.jobs.utils.error_mapper import map_job_error


class JobRepository(IJobRepository):
    def __init__(self, db_manager: IDatabaseManager):
        self._db = db_manager

    def create_job(self, request: JobRequest) -> Job:
        try:
            with self._db.session() as session:
                job_model = JobModel(
                    id=uuid.uuid4(),  # <-- FIXED: Explicit ID
                    job_type=request.job_type.value,
                    source_id=request.source_id,
                    input_params=request.input_params,
                    status=JobStatus.PENDING.value,
                    created_at=datetime.now(timezone.utc)  # <-- FIXED: Explicit timestamp
                )

                session.add(job_model)
                return Job.model_validate(job_model)

        except db_exc.DatabaseError as e:
            raise map_job_error(e)

    def start_job(self, job_id: uuid.UUID) -> Optional[Job]:
        try:
            with self._db.session() as session:
                stmt = (
                    update(JobModel)
                    .where(JobModel.id == job_id)
                    .values(
                        status=JobStatus.PROCESSING.value,
                        started_at=datetime.now(timezone.utc)
                    )
                    .returning(JobModel)
                )

                model = session.execute(stmt).scalar_one_or_none()
                if not model:
                    return None

                return Job.model_validate(model)

        except db_exc.DatabaseError as e:
            raise map_job_error(e)

    def finish_job(self, job_id: uuid.UUID, output_data: dict) -> Job:
        try:
            with self._db.session() as session:
                job_model = session.get(JobModel, job_id)
                if not job_model:
                    raise JobDataError(f"Cannot finish job: Job {job_id} not found.")

                job_model.status = JobStatus.COMPLETED.value
                job_model.output_result = output_data
                job_model.completed_at = datetime.now(timezone.utc)

                session.add(job_model)
                return Job.model_validate(job_model)

        except db_exc.DatabaseError as e:
            raise map_job_error(e)

    def fail_job(self, job_id: uuid.UUID, error_message: str, stack_trace: Optional[str] = None) -> Job:
        try:
            with self._db.session() as session:
                job_model = session.get(JobModel, job_id)
                if not job_model:
                    raise JobDataError(f"Cannot fail job: Job {job_id} not found.")

                job_model.status = JobStatus.FAILED.value
                job_model.error_message = error_message
                job_model.stack_trace = stack_trace
                job_model.completed_at = datetime.now(timezone.utc)

                session.add(job_model)
                return Job.model_validate(job_model)

        except db_exc.DatabaseError as e:
            raise map_job_error(e)

    def cancel_job(self, job_id: uuid.UUID) -> Job:
        try:
            with self._db.session() as session:
                job_model = session.get(JobModel, job_id)
                if not job_model:
                    raise JobDataError(f"Cannot cancel job: Job {job_id} not found.")

                job_model.status = JobStatus.CANCELLED.value
                job_model.completed_at = datetime.now(timezone.utc)

                session.add(job_model)
                return Job.model_validate(job_model)

        except db_exc.DatabaseError as e:
            raise map_job_error(e)

    def get_job_history(self, source_id: uuid.UUID) -> List[Job]:
        try:
            with self._db.session() as session:
                stmt = (
                    select(JobModel)
                    .where(JobModel.source_id == source_id)
                    .order_by(JobModel.created_at.asc())
                )
                models = session.execute(stmt).scalars().all()

                return [Job.model_validate(m) for m in models]

        except db_exc.DatabaseError as e:
            raise map_job_error(e)