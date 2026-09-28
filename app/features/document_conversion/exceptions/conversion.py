# Path: app/features/document_conversion/exceptions/conversion.py

from .base import DocumentConversionError


class UnsupportedDocumentError(DocumentConversionError):
    """The file is not a kind of document this feature converts (see DOCUMENT_KINDS)."""
    pass


class SourceDocumentMissingError(DocumentConversionError):
    """The source has no file record, or its file is gone from the vault."""
    pass


class DocumentReadError(DocumentConversionError):
    """The document could not be opened or parsed: corrupt, truncated, or password-protected."""
    pass


class UndecodableTextError(DocumentConversionError):
    """A text file is in no encoding that could be recognised: indexing it would index mojibake."""
    pass
