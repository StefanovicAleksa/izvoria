# File: app/core/jobs/utils/error_mapper.py

from app.core.database import exceptions as db_exc
from app.core.jobs import exceptions as job_exc


def map_job_error(e: Exception) -> job_exc.JobError:
    """
    Translates Core Database Exceptions into Feature-Specific Job Exceptions.
    """

    # 1. Handle Missing References (e.g., Bad source_id)
    if isinstance(e, db_exc.DBForeignKeyError):
        return job_exc.JobDataError(
            "Cannot process job: The referenced Source does not exist.",
            original_exception=e
        )

    # 2. Handle Data Format Issues
    if isinstance(e, (db_exc.DataError, db_exc.DBNotNullError)):
        return job_exc.JobDataError(
            f"Invalid data provided to Job System: {str(e)}",
            original_exception=e
        )

    # 3. Handle System/Connection Issues (Database down, Network lost)
    if isinstance(e, (db_exc.ConnectionError, db_exc.SystemError, db_exc.DBUnreachableError)):
        return job_exc.JobSystemError(
            "Job System is currently unavailable due to an infrastructure failure.",
            original_exception=e
        )

    # 4. Fallback Catch-all
    return job_exc.JobSystemError(
        f"An unexpected error occurred in the Job System: {str(e)}",
        original_exception=e
    )