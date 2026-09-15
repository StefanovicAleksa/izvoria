# Path: app/features/session_memory/exceptions/base.py

class SessionMemoryError(Exception):
    """
    The Root Exception for the Session Memory feature.
    All other session memory exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
