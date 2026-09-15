# Path: app/features/session_memory/utils/error_mapper.py

from app.core.database import exceptions as db_exc
from app.features.session_memory import exceptions as exc


def map_session_memory_error(e: Exception) -> exc.SessionMemoryError:
    """
    Translates Core Database Exceptions into Feature-Specific Session Memory Exceptions.
    """
    if isinstance(e, exc.SessionMemoryError):
        return e

    # 1. Handle Data Format Issues
    if isinstance(e, (db_exc.DataError, db_exc.DBNotNullError)):
        return exc.SessionMemoryDataError(
            f"Invalid data provided to Session Memory: {str(e)}",
            original_exception=e
        )

    # 2. Handle System/Connection Issues (Database down, Network lost)
    if isinstance(e, (db_exc.ConnectionError, db_exc.SystemError, db_exc.DBUnreachableError)):
        return exc.SessionMemorySystemError(
            "Session Memory is currently unavailable due to an infrastructure failure.",
            original_exception=e
        )

    # 3. Fallback Catch-all
    return exc.SessionMemorySystemError(
        f"An unexpected error occurred in Session Memory: {str(e)}",
        original_exception=e
    )
