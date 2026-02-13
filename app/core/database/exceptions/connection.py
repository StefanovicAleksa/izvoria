from .base import DatabaseError

class ConnectionError(DatabaseError):
    """Base for connectivity issues."""
    pass

class DBConnectionRefusedError(ConnectionError):
    """The database server exists but refused the connection."""
    pass

class DBAuthenticationError(ConnectionError):
    """Wrong username or password."""
    pass

class DBUnreachableError(ConnectionError):
    """Network issue or Database is down/crashed."""
    pass