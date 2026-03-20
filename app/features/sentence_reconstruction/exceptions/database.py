# Canvas: sentence_exceptions_database_v4
# Path: app/features/sentence_reconstruction/exceptions/database.py

from .base import SentenceReconstructionError

class SentenceDatabaseError(SentenceReconstructionError):
    """Generic database failure during Phase 7."""
    pass

class SentencePersistenceError(SentenceDatabaseError):
    """Raised for PostgreSQL Integrity errors (e.g., foreign key constraint fails on source_id)."""
    pass

class SentenceSerializationError(SentenceDatabaseError):
    """Raised when the ReconstructedTranscript fails to serialize into JSONB."""
    pass

class SentenceNotFoundError(SentenceDatabaseError):
    """Raised when attempting to retrieve a transcript that doesn't exist in the database."""
    pass