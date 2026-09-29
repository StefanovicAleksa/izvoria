# Path: app/features/search_history/exceptions/database.py

from .base import SearchHistoryError


class SearchHistoryDataError(SearchHistoryError):
    """Raised when invalid data is provided to the search history."""
    pass


class SearchHistorySystemError(SearchHistoryError):
    """Raised when the search history's underlying database is unavailable."""
    pass
