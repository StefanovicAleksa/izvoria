# Path: app/features/document_conversion/exceptions/conversion.py

from .base import DocumentConversionError


class UnsupportedDocumentError(DocumentConversionError):
    """The file is not a kind of document this feature converts (PDF, DOCX)."""
    pass


class SourceDocumentMissingError(DocumentConversionError):
    """The source has no file record, or its file is gone from the vault."""
    pass


class DocumentReadError(DocumentConversionError):
    """The document could not be opened or parsed: corrupt, truncated, or password-protected."""
    pass
