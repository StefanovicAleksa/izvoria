# Path: app/features/document_conversion/exceptions/__init__.py

from .base import DocumentConversionError
from .conversion import UnsupportedDocumentError, SourceDocumentMissingError, DocumentReadError
from .ocr import OcrError, OcrModelsMissingError, OcrEngineError
from .database import ConversionPersistenceError

__all__ = [
    "DocumentConversionError",
    "UnsupportedDocumentError",
    "SourceDocumentMissingError",
    "DocumentReadError",
    "OcrError",
    "OcrModelsMissingError",
    "OcrEngineError",
    "ConversionPersistenceError",
]
