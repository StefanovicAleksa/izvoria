from .base import AsrCorrectionError

class AsrCorrectionDatabaseError(AsrCorrectionError):
    """Generic database failure during the correction pipeline."""
    pass

class GlossaryPersistenceError(AsrCorrectionDatabaseError):
    """Raised when the extracted Domain Glossary fails to save to Postgres."""
    pass

class AuditTrailSaveError(AsrCorrectionDatabaseError):
    """Raised if the Before/After audit records fail to persist, violating our data integrity guarantees."""
    pass