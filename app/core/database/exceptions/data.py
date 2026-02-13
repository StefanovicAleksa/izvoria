from .base import DatabaseError

class DataError(DatabaseError):
    """Base for invalid data format issues."""
    pass

class DBNotNullError(DataError):
    """Tried to insert NULL into a required field."""
    pass

class DBInvalidTextRepresentation(DataError):
    """Invalid input syntax for type (e.g. text into integer column)."""
    pass