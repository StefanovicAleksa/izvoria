from .base import StorageError

class ResourceNotFound(StorageError):
    """Base class for missing resources."""
    pass

class SourceNotFound(ResourceNotFound):
    """Raised when a specific Source ID or Hash cannot be found."""
    pass

class FileNotFound(ResourceNotFound):
    """Raised when a specific File record cannot be found."""
    pass