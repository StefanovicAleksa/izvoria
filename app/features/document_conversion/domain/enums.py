# Path: app/features/document_conversion/domain/enums.py

from enum import Enum


class DocumentKind(str, Enum):
    PDF = "pdf"
    DOCX = "docx"


class ExtractionMethod(str, Enum):
    """How a page's text was obtained."""
    # The PDF's own text layer, via MarkItDown: exact
    TEXT_LAYER = "text_layer"
    # Recognised from the rendered page by OCR: may contain misread characters
    OCR = "ocr"
    # No text: a scanned page on an appliance without OCR, or a page OCR found nothing on
    NONE = "none"
