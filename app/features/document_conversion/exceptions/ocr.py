# Path: app/features/document_conversion/exceptions/ocr.py

from .base import DocumentConversionError


class OcrError(DocumentConversionError):
    """Base exception for scanned-page recognition."""
    pass


class OcrModelsMissingError(OcrError):
    """The OCR model files are not in DOC_OCR_MODEL_DIR. The appliance never downloads them at runtime."""
    pass


class OcrEngineError(OcrError):
    """The OCR engine failed on a page (onnxruntime error, unreadable image)."""
    pass
