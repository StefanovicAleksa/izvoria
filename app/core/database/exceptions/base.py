class DatabaseError(Exception):
    """
    The Root Exception for the entire Onyx Database Layer.
    All other database exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None, details: dict = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.details = details or {}