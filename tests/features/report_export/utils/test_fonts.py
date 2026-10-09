# Path: tests/features/report_export/utils/test_fonts.py

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.features.report_export.exceptions.rendering import FontMissingError
from app.features.report_export.utils import fonts


class TestRegister:
    def test_every_face_and_the_symbol_fallback_are_added(self):
        pdf = MagicMock()
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(fonts, "font_dir", return_value=Path("/fonts")), \
                patch.object(fonts, "symbols_font", return_value=Path("/zia/STIXTwoMath-Regular.ttf")):
            fonts.register(pdf)

        added = [call.args for call in pdf.add_font.call_args_list]
        assert (fonts.SERIF, "", "/fonts/SourceSerif4-Regular.ttf") in added
        assert (fonts.SANS, "B", "/fonts/Inter-SemiBold.ttf") in added
        assert added[-1] == (fonts.SYMBOLS, "", "/zia/STIXTwoMath-Regular.ttf")
        pdf.set_fallback_fonts.assert_called_once_with([fonts.SYMBOLS], exact_match=False)

    def test_a_missing_font_refuses_the_export(self):
        with patch.object(Path, "is_file", return_value=False), \
                patch.object(fonts, "font_dir", return_value=Path("/fonts")):
            with pytest.raises(FontMissingError, match="SourceSerif4-Regular.ttf"):
                fonts.register(MagicMock())

    def test_the_fonts_live_in_the_repository_data_folder(self):
        assert fonts.font_dir().parts[-2:] == ("data", "fonts")
