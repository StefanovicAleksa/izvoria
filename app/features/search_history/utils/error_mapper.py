# Path: app/features/search_history/utils/error_mapper.py

from pydantic import ValidationError

from app.core.database import exceptions as db_exc
from app.features.search_history import exceptions as exc


def map_search_history_error(e: Exception) -> exc.SearchHistoryError:
    """
    Translates Core Database Exceptions into Feature-Specific Search History Exceptions.
    """
    if isinstance(e, exc.SearchHistoryError):
        return e

    # 1. Handle Data Format Issues (including a record whose scores don't line up with its chunks)
    if isinstance(e, (db_exc.DataError, db_exc.DBNotNullError, ValidationError, ValueError)):
        return exc.SearchHistoryDataError(
            f"Invalid data provided to Search History: {str(e)}",
            original_exception=e
        )

    # 2. Handle System/Connection Issues (Database down, Network lost)
    if isinstance(e, (db_exc.ConnectionError, db_exc.SystemError, db_exc.DBUnreachableError)):
        return exc.SearchHistorySystemError(
            "Search History is currently unavailable due to an infrastructure failure.",
            original_exception=e
        )

    # 3. Fallback Catch-all
    return exc.SearchHistorySystemError(
        f"An unexpected error occurred in Search History: {str(e)}",
        original_exception=e
    )
