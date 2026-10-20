"""The window icon is drawn from the mark in data/brand, and its absence never stops the app."""

from izvoria_desktop.ui import icons


class TestAppIcon:
    def test_the_shipped_mark_gives_every_size(self, qt_app):
        icon = icons.app_icon()

        assert not icon.isNull()
        assert {size.width() for size in icon.availableSizes()} == set(icons.APP_ICON_SIZES)

    def test_a_missing_file_gives_the_default_icon(self, qt_app, tmp_path):
        assert icons.app_icon(tmp_path / "missing.svg").isNull()

    def test_an_unreadable_file_gives_the_default_icon(self, qt_app, tmp_path):
        broken = tmp_path / "broken.svg"
        broken.write_text("not an svg")
        assert icons.app_icon(broken).isNull()
