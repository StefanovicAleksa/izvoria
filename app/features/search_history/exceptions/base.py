# Path: app/features/search_history/exceptions/base.py

class SearchHistoryError(Exception):
    """
    The Root Exception for the Search History feature.
    All other search history exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
