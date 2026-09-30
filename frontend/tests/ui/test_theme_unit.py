# Canvas: desktop_test_theme_v3
# Path: frontend/tests/ui/test_theme_unit.py

import re
from pathlib import Path

import pytest

from onyx_desktop.ui import theme


@pytest.fixture(autouse=True)
def _restore_default():
    yield
    theme.use()


class TestTheme:
    @pytest.mark.parametrize("mode", theme.MODES)
    def test_every_mode_builds_a_complete_stylesheet(self, mode):
        sheet = theme.build_stylesheet(theme.PALETTES[mode])

        # An unformatted field would leave a literal {name} behind, and Qt ignores a malformed rule silently
        assert re.search(r"\{[a-z_.]+\}", sheet) is None
        assert sheet.count("{") == sheet.count("}")

    def test_use_rebinds_the_names_widgets_read(self):
        theme.use("light")

        assert theme.MODE == "light"
        assert theme.PALETTE is theme.LIGHT
        assert theme.CITATION == theme.LIGHT.citation
        assert theme.STATUS_COLOURS["error"] == theme.LIGHT.danger
        assert theme.STYLESHEET == theme.build_stylesheet(theme.LIGHT)

    def test_dark_is_the_default(self):
        theme.use()
        assert theme.MODE == "dark"
        assert theme.PALETTE is theme.DARK

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="mode 'sepia'"):
            theme.use("sepia")

    def test_the_arrow_assets_the_stylesheet_points_at_exist(self):
        for url in re.findall(r"url\(([^)]+)\)", theme.STYLESHEET):
            assert Path(url).is_file(), url
