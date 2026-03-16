# Canvas: domain_adaptation_exceptions_database_v1
# Path: app/features/domain_adaptation/exceptions/database.py

from .base import DomainAdaptationError

class DomainDatabaseError(DomainAdaptationError):
    """Generic database failure during Domain Adaptation persistence."""
    pass

class GlossaryPersistenceError(DomainDatabaseError):
    """Raised when the final Domain Glossary fails to save to PostgreSQL."""
    pass