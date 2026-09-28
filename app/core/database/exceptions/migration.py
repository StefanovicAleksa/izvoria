from .system import SystemError

class DBMigrationError(SystemError):
    """Bringing the schema up to date failed. Postgres DDL is transactional, so the database is left as it was."""
    pass
