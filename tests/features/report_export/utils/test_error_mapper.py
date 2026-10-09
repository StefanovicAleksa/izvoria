# Path: tests/features/report_export/utils/test_error_mapper.py

import pytest

from app.features.report_export import exceptions as exc
from app.features.report_export.utils.error_mapper import map_export_error


class FPDFException(Exception):
    """Stands in for fpdf.errors.FPDFException, which the mapper recognises by name."""


class TestMapExportError:
    @pytest.mark.parametrize("raised, expected", [
        (FileNotFoundError("Inter-Regular.ttf"), exc.FontMissingError),
        (PermissionError("denied"), exc.FontMissingError),
        (FPDFException("Undefined font"), exc.DocumentRenderError),
        (ValueError("bad layout"), exc.DocumentRenderError),
    ])
    def test_low_level_errors_become_typed_export_errors(self, raised, expected):
        mapped = map_export_error(raised, context="Exporting")

        assert type(mapped) is expected
        assert mapped.original_exception is raised
        assert str(mapped).startswith("Exporting: ")

    def test_an_export_error_passes_through_unchanged(self):
        error = exc.MathRenderError("x")

        assert map_export_error(error) is error
