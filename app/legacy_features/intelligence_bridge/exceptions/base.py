class IntelligenceBridgeError(Exception):
    """Root exception for the Intelligence Bridge workflow."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception