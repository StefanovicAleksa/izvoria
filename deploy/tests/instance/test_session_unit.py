import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deploy.instance.agent.config import AgentSettings
from deploy.instance.agent.session import (
    AppSession, UnknownLanguage, launch_command, launch_env,
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

    @pytest.mark.parametrize("language", ["sr", "en"])
    def test_env_selects_the_starting_language_and_the_one_database(self, settings, language):
        env = launch_env(settings, language, {"PATH": "/usr/bin", "APPLIANCE_LANGUAGE": "de"})
        assert env == {
            "PATH": "/usr/bin",
            "ENV": "production",
            "APPLIANCE_LANGUAGE": language,
            "POSTGRES_DB": "onyx",
            "DISPLAY": ":1",
            "QT_QPA_PLATFORM": "xcb",
        }

    def test_env_hands_the_app_the_offer_endpoint_and_its_token(self, settings):
        env = launch_env(settings, "en", {}, offer_token="tok")
        assert env["ONYX_SOURCE_OFFER_URL"] == "http://127.0.0.1:8090/offers"
        assert env["ONYX_SOURCE_OFFER_TOKEN"] == "tok"
        assert "AGENT_SECRET" not in env

    def test_env_rejects_a_language_without_a_pack(self, settings):
        with pytest.raises(UnknownLanguage):
            launch_env(settings, "de", {})


class TestEnsure:
    def test_launches_and_records_the_app_when_none_runs(self, settings, popen):
        with patch("deploy.instance.agent.session._is_alive", return_value=True):
            app = AppSession(settings, popen=popen).ensure("sr")

        assert (app.pid, app.language) == (4242, "sr")
        args, kwargs = popen.call_args
        assert args[0] == launch_command(settings)
        assert kwargs["cwd"] == "/workspace/onyx_v3"
        assert kwargs["env"]["POSTGRES_DB"] == "onyx"
        assert kwargs["start_new_session"] is True
        assert kwargs["stdout"].name == str(settings.log_dir / "app.log")
        assert kwargs["stderr"] is subprocess.STDOUT
        state = json.loads(settings.state_file.read_text())
        assert {k: state[k] for k in ("pid", "language")} == {"pid": 4242, "language": "sr"}
        # The app got the same per-launch token the agent will check, and the agent secret stays with the agent
        assert len(state["offer_token"]) >= 40
        assert kwargs["env"]["ONYX_SOURCE_OFFER_TOKEN"] == state["offer_token"]
        assert settings.state_file.stat().st_mode & 0o077 == 0

    def test_a_new_launch_gets_a_new_token(self, settings, popen):
        with patch("deploy.instance.agent.session._is_alive", return_value=False):
            AppSession(settings, popen=popen).ensure("sr")
            first = json.loads(settings.state_file.read_text())["offer_token"]
            AppSession(settings, popen=popen).ensure("sr")
        assert json.loads(settings.state_file.read_text())["offer_token"] != first

    def test_offer_token_is_read_back_for_the_running_app(self, settings, popen):
        with patch("deploy.instance.agent.session._is_alive", return_value=True):
            session = AppSession(settings, popen=popen)
            session.ensure("sr")
            assert session.offer_token() == json.loads(settings.state_file.read_text())["offer_token"]

    def test_no_app_means_no_token(self, settings, popen):
        write_state(settings, 99, "sr")
        with patch("deploy.instance.agent.session._is_alive", return_value=False):
            assert AppSession(settings, popen=popen).offer_token() is None

    def test_an_app_from_before_offers_has_no_token(self, settings, popen):
        write_state(settings, 99, "sr")
        with patch("deploy.instance.agent.session._is_alive", return_value=True):
            assert AppSession(settings, popen=popen).offer_token() is None

    @pytest.mark.parametrize("language", ["en", "sr"])
    def test_a_running_app_is_kept_whatever_language_is_asked_for(self, settings, popen, language):
        # It switches languages itself, without losing a job it may be running
        write_state(settings, 99, "en")
        with patch("deploy.instance.agent.session._is_alive", return_value=True), \
                patch("deploy.instance.agent.session._terminate") as terminate:
            app = AppSession(settings, popen=popen).ensure(language)

        assert (app.pid, app.language) == (99, "en")
        terminate.assert_not_called()
        popen.assert_not_called()

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
