import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deploy.instance.agent.config import AgentSettings
from deploy.instance.agent.session import (
    AppSession, SessionBusy, UnknownLanguage, launch_command, launch_env,
)


@pytest.fixture
def settings(tmp_path):
    return AgentSettings(
        secret="s" * 32,
        onyx_root=Path("/workspace/onyx_v3"),
        upload_root=tmp_path / "uploads",
        state_file=tmp_path / "state" / "app.json",
        log_dir=tmp_path / "logs",
    )


@pytest.fixture
def popen():
    return MagicMock(return_value=MagicMock(pid=4242))


def write_state(settings, pid, language):
    settings.state_file.parent.mkdir(parents=True, exist_ok=True)
    settings.state_file.write_text(json.dumps({"pid": pid, "language": language}))


class TestLaunch:
    def test_command_boots_the_production_appliance_without_a_click(self, settings):
        assert launch_command(settings) == [
            "/workspace/onyx_v3/venv/bin/python", "frontend/main.py", "--env", "production", "--autostart",
        ]

    @pytest.mark.parametrize("language, database", [("sr", "onyx_sr"), ("sl", "onyx_sl")])
    def test_env_selects_the_language_and_its_database(self, settings, language, database):
        env = launch_env(settings, language, {"PATH": "/usr/bin", "APPLIANCE_LANGUAGE": "en"})
        assert env == {
            "PATH": "/usr/bin",
            "ENV": "production",
            "APPLIANCE_LANGUAGE": language,
            "POSTGRES_DB": database,
            "DISPLAY": ":1",
            "QT_QPA_PLATFORM": "xcb",
        }

    def test_env_rejects_a_language_without_a_database(self, settings):
        with pytest.raises(UnknownLanguage):
            launch_env(settings, "en", {})


class TestEnsure:
    def test_launches_and_records_the_app_when_none_runs(self, settings, popen):
        with patch("deploy.instance.agent.session._is_alive", return_value=True):
            app = AppSession(settings, popen=popen).ensure("sr", busy=False)

        assert (app.pid, app.language) == (4242, "sr")
        args, kwargs = popen.call_args
        assert args[0] == launch_command(settings)
        assert kwargs["cwd"] == "/workspace/onyx_v3"
        assert kwargs["env"]["POSTGRES_DB"] == "onyx_sr"
        assert kwargs["start_new_session"] is True
        assert kwargs["stdout"].name == str(settings.log_dir / "app-sr.log")
        assert kwargs["stderr"] is subprocess.STDOUT
        assert json.loads(settings.state_file.read_text()) == {"pid": 4242, "language": "sr"}

    def test_same_language_is_a_no_op(self, settings, popen):
        write_state(settings, 99, "sl")
        with patch("deploy.instance.agent.session._is_alive", return_value=True):
            app = AppSession(settings, popen=popen).ensure("sl", busy=True)

        assert (app.pid, app.language) == (99, "sl")
        popen.assert_not_called()

    def test_refuses_to_switch_language_while_busy(self, settings, popen):
        write_state(settings, 99, "sl")
        with patch("deploy.instance.agent.session._is_alive", return_value=True), \
                patch("deploy.instance.agent.session._terminate") as terminate:
            with pytest.raises(SessionBusy):
                AppSession(settings, popen=popen).ensure("sr", busy=True)

        terminate.assert_not_called()
        popen.assert_not_called()

    def test_switching_language_when_idle_stops_the_old_app_first(self, settings, popen):
        write_state(settings, 99, "sl")
        with patch("deploy.instance.agent.session._is_alive", return_value=True), \
                patch("deploy.instance.agent.session._terminate") as terminate:
            app = AppSession(settings, popen=popen).ensure("sr", busy=False)

        terminate.assert_called_once()
        assert terminate.call_args.args[0] == 99
        assert app.language == "sr"

    def test_a_dead_pid_in_the_state_file_counts_as_no_app(self, settings, popen):
        write_state(settings, 99, "sr")
        with patch("deploy.instance.agent.session._is_alive", return_value=False):
            assert AppSession(settings, popen=popen).current() is None


class TestStop:
    def test_terminates_and_forgets_the_app(self, settings, popen):
        write_state(settings, 99, "sr")
        with patch("deploy.instance.agent.session._is_alive", return_value=True), \
                patch("deploy.instance.agent.session._terminate") as terminate:
            AppSession(settings, popen=popen).stop()

        assert terminate.call_args.args[0] == 99
        assert not settings.state_file.exists()

    def test_nothing_running_is_fine(self, settings, popen):
        AppSession(settings, popen=popen).stop()
