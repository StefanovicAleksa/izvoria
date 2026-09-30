# Path: frontend/tests/ui/test_history_list_unit.py

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from onyx_desktop.ui.widgets.history_list import HistoryEntry, HistoryList  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def pane(qt_app):
    widget = HistoryList("Past searches", "Nothing yet.", deletable=True, delete_label="Delete search")
    yield widget
    widget.deleteLater()


ENTRIES = [HistoryEntry(key="a", title="first", subtitle="today", badge="Verified"),
           HistoryEntry(key="b", title="second")]


class TestHistoryList:
    def test_starts_as_unavailable_until_given_entries(self, pane):
        assert pane.keys() == []
        assert "Start the appliance" in pane._placeholder.text()

    def test_lists_entries_in_order_with_badge_and_subtitle(self, pane):
        pane.set_entries(ENTRIES)
        assert pane.keys() == ["a", "b"]
        assert pane._list.item(0).text() == "first\nVerified · today"
        assert pane._list.item(1).text() == "second"
        assert pane._placeholder.isHidden()

    def test_empty_shows_the_empty_text(self, pane):
        pane.set_entries([])
        assert pane._placeholder.text() == "Nothing yet." and not pane._placeholder.isHidden()

    def test_a_click_emits_the_key(self, pane):
        pane.set_entries(ENTRIES)
        chosen = []
        pane.itemChosen.connect(chosen.append)
        pane._list.itemClicked.emit(pane._list.item(1))
        assert chosen == ["b"]

    def test_select_highlights_without_emitting(self, pane):
        pane.set_entries(ENTRIES)
        chosen = []
        pane.itemChosen.connect(chosen.append)
        pane.select("b")
        assert pane.current_key() == "b" and chosen == []

    def test_refreshing_keeps_the_selection(self, pane):
        pane.set_entries(ENTRIES)
        pane.select("b")
        pane.set_entries(ENTRIES + [HistoryEntry(key="c", title="third")])
        assert pane.current_key() == "b"

    @pytest.mark.parametrize("answer, expected", [
        (QMessageBox.StandardButton.Yes, ["a"]), (QMessageBox.StandardButton.No, []),
    ])
    def test_delete_asks_first(self, pane, monkeypatch, answer, expected):
        pane.set_entries(ENTRIES)
        pane.select("a")
        monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: answer)
        deleted = []
        pane.deleteRequested.connect(deleted.append)
        pane._delete.click()
        assert deleted == expected

    def test_delete_needs_a_selection(self, pane):
        pane.set_entries(ENTRIES)
        assert not pane._delete.isEnabled()
