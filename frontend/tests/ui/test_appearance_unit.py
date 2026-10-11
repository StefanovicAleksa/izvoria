# Canvas: desktop_test_appearance_v1
# Path: frontend/tests/ui/test_appearance_unit.py

from PySide6.QtCore import QSettings

from izvoria_desktop.ui import appearance


class TestAppearance:
    def test_nothing_is_saved_until_the_user_switches(self):
        assert appearance.saved_mode() is None

    def test_a_saved_mode_comes_back(self):
        appearance.save_mode("light")
        assert appearance.saved_mode() == "light"

    def test_a_value_that_is_no_longer_a_mode_is_ignored(self):
        QSettings("Onyx", "Onyx Desktop").setValue("appearance/mode", "sepia")
        assert appearance.saved_mode() is None

    def test_a_saved_display_language_comes_back(self):
        assert appearance.saved_display_language() is None
        appearance.save_display_language("sr")
        assert appearance.saved_display_language() == "sr"

    def test_a_display_language_with_no_catalog_is_ignored(self):
        QSettings("Onyx", "Onyx Desktop").setValue("appearance/display_language", "xx")
        assert appearance.saved_display_language() is None
