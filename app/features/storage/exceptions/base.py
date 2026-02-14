class StorageError(Exception):
    """
    Base exception for all Storage-related errors.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception