# Path: app/features/document_conversion/exceptions/database.py

from .base import DocumentConversionError


class ConversionPersistenceError(DocumentConversionError):
    """A converted document could not be saved to or read from PostgreSQL."""
    pass
