# File: tests/core/jobs/test_error_mapper.py

from app.core.database import exceptions as db_exc
from app.core.jobs import exceptions as job_exc
from app.core.jobs.utils.error_mapper import map_job_error

class TestJobErrorMapper:
    def test_map_foreign_key_error(self):
        """DBForeignKeyError -> JobDataError"""
        e = db_exc.DBForeignKeyError("Missing Parent")
        mapped = map_job_error(e)
        assert isinstance(mapped, job_exc.JobDataError)
        assert "referenced Source does not exist" in str(mapped)

    def test_map_data_error(self):
        """DataError -> JobDataError"""
        e = db_exc.DataError("Invalid UUID")
        mapped = map_job_error(e)
        assert isinstance(mapped, job_exc.JobDataError)
        assert "Invalid data provided" in str(mapped)

    def test_map_system_error(self):
        """SystemError -> JobSystemError"""
        e = db_exc.SystemError("DB offline")
        mapped = map_job_error(e)
        assert isinstance(mapped, job_exc.JobSystemError)
        assert "infrastructure failure" in str(mapped)

    def test_map_unknown_error(self):
        """Exception -> JobSystemError (Fallback)"""
        e = Exception("Something totally unexpected")
        mapped = map_job_error(e)
        assert isinstance(mapped, job_exc.JobSystemError)
        assert "unexpected error" in str(mapped)