from .base import DatabaseError

class IntegrityError(DatabaseError):
    """Base for violations of database constraints."""
    pass

class DBUniqueError(IntegrityError):
    """
    Raised when a Unique Constraint is violated.
    Example: Trying to add a Source with a file_hash that already exists.
    """
    pass

class DBForeignKeyError(IntegrityError):
    """
    Raised when a Foreign Key is missing.
    Example: Trying to add a File to a Source ID that doesn't exist.
    """
    pass

class DBCheckViolationError(IntegrityError):
    """
    Raised when a value violates a specific check constraint.
    Example: 'age' must be > 18.
    """
    pass