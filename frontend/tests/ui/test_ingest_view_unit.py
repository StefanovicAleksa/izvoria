# Path: frontend/tests/ui/test_ingest_view_unit.py

import pytest

pytest.importorskip("PySide6")

from onyx_desktop.domain.entities import LanguageOption  # noqa: E402
from onyx_desktop.domain.enums import IngestMode  # noqa: E402
from onyx_desktop.ui.views.ingest_view import IngestView  # noqa: E402

pytestmark = pytest.mark.gui

LANGUAGES = [LanguageOption("en", "English"), LanguageOption("sr", "Srpski")]


@pytest.fixture
def view(qt_app):
    instance = IngestView()
    instance.set_languages(LANGUAGES, "en")
    yield instance
    instance.deleteLater()


class TestFileLanguage:
    def _request(self, view):
        requests = []
        view.ingestRequested.connect(lambda *args: requests.append(args))
        view._path.setText("/media/batch")
        view._on_start()
        return requests

    def test_the_request_carries_the_language_as_a_plain_code(self, view):
        view._language.setCurrentIndex(view._language.findData("sr"))

        [(path, mode, language)] = self._request(view)

        assert (path, mode, language) == ("/media/batch", IngestMode.VIDEO, "sr")
        # Rebuilt at the boundary: a QVariant must never reach the gateway (the QComboBox gotcha)
        assert type(language) is str and type(mode) is IngestMode

    def test_defaults_to_the_active_language_and_follows_a_switch(self, view):
        assert self._request(view)[0][2] == "en"

        view.select_language("sr")

        assert self._request(view)[0][2] == "sr"

    def test_languages_are_listed_by_their_own_names(self, view):
        assert [view._language.itemText(i) for i in range(view._language.count())] == [
            "English", "Srpski"]

    def test_the_picker_is_locked_while_busy(self, view):
        view.set_busy(True)
        assert not view._language.isEnabled()
        view.set_busy(False)
        assert view._language.isEnabled()
