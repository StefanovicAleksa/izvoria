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

class SourceLanguageConflict(ResourceConflict):
    """
    Raised when a file already ingested in one language is ingested again in another. Its transcript and every
    stage after it were produced in the first language, and the language check vouched for that language.
    """
    pass

class FileAlreadyExists(ResourceConflict):
    """Raised when trying to add a duplicate file path."""
    pass