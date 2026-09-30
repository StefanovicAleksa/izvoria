from unittest.mock import MagicMock, patch

import pytest

from onyx_desktop.service import desktop_app


@pytest.fixture
def wiring():
    bridge = MagicMock()
    app = MagicMock()
    app.exec.return_value = 0
    with patch("PySide6.QtWidgets.QApplication", return_value=app), \
            patch("onyx_desktop.service.engine_bridge.EngineBridge", return_value=bridge), \
            patch("onyx_desktop.ui.main_window.MainWindow") as window_cls, \
            patch("onyx_desktop.service.log_relay.LogRelay"), \
            patch.object(desktop_app, "build_gateway"), \
            patch.object(desktop_app, "configure_logging"), \
            patch.object(desktop_app, "_assert_single_qt"):
        yield bridge, window_cls.return_value


class TestRun:
    def test_autostart_boots_once_after_the_window_is_shown(self, wiring):
        bridge, window = wiring
        order = []
        window.show.side_effect = lambda: order.append("show")
        bridge.boot.side_effect = lambda: order.append("boot")

        assert desktop_app.run(["onyx"], autostart=True) == 0

        bridge.boot.assert_called_once_with()
        assert order == ["show", "boot"]

    def test_without_autostart_waits_for_the_start_button(self, wiring):
        bridge, _window = wiring

        desktop_app.run(["onyx"])

        bridge.boot.assert_not_called()

    def test_the_flag_picks_the_mode(self, wiring):
        with patch("onyx_desktop.ui.theme.use") as use:
            desktop_app.run(["onyx"], mode="light")
        use.assert_called_once_with("light")

    def test_the_environment_picks_it_when_nothing_else_does(self, wiring, monkeypatch):
        monkeypatch.setenv("ONYX_MODE", "light")
        with patch("onyx_desktop.ui.theme.use") as use:
            desktop_app.run(["onyx"])
        use.assert_called_once_with("light")

    def test_the_default_is_dark(self, wiring, monkeypatch):
        monkeypatch.delenv("ONYX_MODE", raising=False)
        with patch("onyx_desktop.ui.theme.use") as use:
            desktop_app.run(["onyx"])
        use.assert_called_once_with("dark")

    def test_the_mode_picked_in_the_app_beats_the_environment(self, wiring, monkeypatch):
        from onyx_desktop.ui import appearance

        monkeypatch.setenv("ONYX_MODE", "dark")
        appearance.save_mode("light")
        with patch("onyx_desktop.ui.theme.use") as use:
            desktop_app.run(["onyx"])
        use.assert_called_once_with("light")

    def test_an_explicit_flag_beats_the_mode_picked_in_the_app(self, wiring):
        from onyx_desktop.ui import appearance

        appearance.save_mode("light")
        with patch("onyx_desktop.ui.theme.use") as use:
            desktop_app.run(["onyx"], mode="dark")
        use.assert_called_once_with("dark")
