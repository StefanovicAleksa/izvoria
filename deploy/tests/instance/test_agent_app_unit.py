from pathlib import Path
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from deploy.instance.agent.app import SECRET_HEADER, create_app
from deploy.instance.agent.config import AgentSettings
from deploy.instance.agent.session import RunningApp, UnknownLanguage

SECRET = "k" * 40
AUTH = {SECRET_HEADER: SECRET}
BATCH = "20260928-153000"


@pytest.fixture
def settings(tmp_path):
    return AgentSettings(secret=SECRET, onyx_root=Path("/x"), upload_root=tmp_path / "up",
                         state_file=tmp_path / "app.json", log_dir=tmp_path / "logs")


@pytest.fixture
def session():
    return MagicMock()


@pytest.fixture
def sampler():
    return MagicMock(busy=False)


@pytest.fixture
def client(settings, session, sampler):
    app = create_app(settings, session=session, sampler=sampler, postgres_check=lambda: True,
                     display_check=lambda display: display == ":1", sample_in_background=False)
    with TestClient(app) as c:
        yield c


class TestAuth:
    @pytest.mark.parametrize("headers", [{}, {SECRET_HEADER: "wrong"}])
    @pytest.mark.parametrize("method, path", [
        ("get", "/health"), ("post", "/session"), ("post", "/shutdown"), ("put", f"/upload/sr/{BATCH}/a.mp4"),
    ])
    def test_every_route_needs_the_secret(self, client, session, headers, method, path):
        assert getattr(client, method)(path, headers=headers).status_code == 401
        session.ensure.assert_not_called()
        session.stop.assert_not_called()


class TestHealth:
    def test_reports_every_readiness_signal(self, client, session, sampler):
        session.current.return_value = RunningApp(pid=7, language="en")
        sampler.busy = True
        assert client.get("/health", headers=AUTH).json() == {
            "postgres": True, "display": True, "app": {"pid": 7, "language": "en"}, "busy": True,
        }

    def test_no_app_yet(self, client, session):
        session.current.return_value = None
        assert client.get("/health", headers=AUTH).json()["app"] is None


class TestSession:
    def test_passes_the_starting_language(self, client, session, sampler):
        session.ensure.return_value = RunningApp(pid=7, language="sr")
        response = client.post("/session", json={"language": "sr"}, headers=AUTH)
        assert response.json() == {"pid": 7, "language": "sr"}
        session.ensure.assert_called_once_with("sr")

    def test_an_unknown_language_is_a_bad_request(self, client, session):
        session.ensure.side_effect = UnknownLanguage("de")
        assert client.post("/session", json={"language": "de"}, headers=AUTH).status_code == 400


class TestShutdown:
    def test_stops_the_app(self, client, session):
        assert client.post("/shutdown", headers=AUTH).json() == {"stopped": True}
        session.stop.assert_called_once_with()


class TestUpload:
    def test_streams_the_body_into_the_batch_folder(self, client, settings):
        response = client.put(f"/upload/sr/{BATCH}/Hearing 1.mp4", content=b"abc" * 1000, headers=AUTH)
        assert response.status_code == 201
        folder = (settings.upload_root / "sr" / BATCH).resolve()
        assert response.json() == {"folder": str(folder), "name": "Hearing 1.mp4", "bytes": 3000}
        assert (folder / "Hearing 1.mp4").read_bytes() == b"abc" * 1000
        assert not list(folder.glob("*.part"))

    @pytest.mark.parametrize("path", [
        f"/upload/de/{BATCH}/a.mp4",
        f"/upload/sr/notabatch/a.mp4",
        f"/upload/sr/{BATCH}/a.exe",
        f"/upload/sr/{BATCH}/..%2F..%2Fa.mp4",
    ])
    def test_rejects_bad_targets(self, client, settings, path):
        response = client.put(path, content=b"x", headers=AUTH)
        assert response.status_code in (400, 404)
        escaped = [p for p in settings.upload_root.parent.rglob("a.mp4")
                   if not p.is_relative_to(settings.upload_root)]
        assert escaped == []
