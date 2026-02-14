from .base import StorageError

class ResourceConflict(StorageError):
    """Base class for resource conflicts (duplicates)."""
    pass

class SourceAlreadyExists(ResourceConflict):
    """
    Raised when trying to add a Source that already exists
    (e.g., same file hash).
    """
    pass

class FileAlreadyExists(ResourceConflict):
    """Raised when trying to add a duplicate file path."""
    pass