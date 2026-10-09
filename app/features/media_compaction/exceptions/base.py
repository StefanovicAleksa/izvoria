class MediaCompactionError(Exception):
    """
    The Root Exception for the Media Compaction feature.
    All other compaction exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
