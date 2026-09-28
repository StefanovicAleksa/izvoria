# Path: app/features/document_conversion/utils/error_mapper.py

from markitdown import FileConversionException, UnsupportedFormatException
from pypdfium2 import PdfiumError
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.document_conversion import exceptions as exc


def _is_onnxruntime_error(e: Exception) -> bool:
    return type(e).__module__.startswith("onnxruntime")


def map_conversion_error(e: Exception, context: str = "") -> exc.DocumentConversionError:
    """Translates pypdfium2, MarkItDown, onnxruntime, filesystem and database errors into this feature's types."""
    if isinstance(e, exc.DocumentConversionError):
        return e

    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        return exc.ConversionPersistenceError(
            f"Database operation failed in Document Conversion: {e} | Context: {context}", original_exception=e
        )

    if isinstance(e, FileNotFoundError):
        return exc.SourceDocumentMissingError(
            f"Document file not found: {e} | Context: {context}", original_exception=e
        )

    if isinstance(e, PdfiumError):
        # pypdfium2 reports a password-protected file as "Incorrect password error"
        reason = "password-protected" if "password" in str(e).lower() else "corrupt or unreadable"
        return exc.DocumentReadError(
            f"PDF is {reason}: {e} | Context: {context}", original_exception=e
        )

    if isinstance(e, (FileConversionException, UnsupportedFormatException)):
        return exc.DocumentReadError(
            f"MarkItDown could not convert the document: {e} | Context: {context}", original_exception=e
        )

    if _is_onnxruntime_error(e):
        return exc.OcrEngineError(
            f"OCR engine failed: {e} | Context: {context}", original_exception=e
        )

    return exc.DocumentConversionError(
        f"Unexpected failure in Document Conversion: {e} | Context: {context}", original_exception=e
    )
