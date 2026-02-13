from .base import DatabaseError

class SystemError(DatabaseError):
    """Base for internal DB system failures."""
    pass

class DBTransactionError(SystemError):
    """The transaction failed or was rolled back unexpectedly."""
    pass

class DBSchemaError(SystemError):
    """The table or column does not exist (Migration issue)."""
    pass