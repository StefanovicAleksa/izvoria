# Path: tests/features/document_conversion/utils/test_error_mapper.py

import pytest
from markitdown import FileConversionException
from pypdfium2 import PdfiumError
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.document_conversion import exceptions as exc
from app.features.document_conversion.utils.error_mapper import map_conversion_error


class TestConversionErrorMapper:
    @pytest.mark.parametrize("error, expected_type", [
        (SQLAlchemyError("down"), exc.ConversionPersistenceError),
        (core_db_exc.DatabaseError("down"), exc.ConversionPersistenceError),
        (FileNotFoundError("gone.pdf"), exc.SourceDocumentMissingError),
        (PdfiumError("Failed to load document (PDFium: Incorrect password error)."), exc.DocumentReadError),
        (PdfiumError("Failed to load document (PDFium: Data format error)."), exc.DocumentReadError),
        (FileConversionException("bad docx"), exc.DocumentReadError),
        (RuntimeError("anything else"), exc.DocumentConversionError),
    ])
    def test_translates_to_feature_exceptions(self, error, expected_type):
        mapped = map_conversion_error(error, context="test")

        assert type(mapped) is expected_type
        assert mapped.original_exception is error

    def test_names_a_password_protected_pdf(self):
        mapped = map_conversion_error(PdfiumError("Failed to load document (PDFium: Incorrect password error)."))

        assert "password-protected" in str(mapped)

    def test_does_not_double_wrap(self):
        original = exc.OcrEngineError("already mapped")

        assert map_conversion_error(original) is original
