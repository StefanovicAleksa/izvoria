from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from deploy.gateway import app as gateway_app
from deploy.gateway.agent_client import AgentHealth
from deploy.gateway.config import GatewaySettings
from deploy.gateway.proxy import websockify_url
from deploy.gateway.vast_client import FakeVastClient

PASSWORD = "correct horse battery"


@pytest.fixture
def settings():
    return GatewaySettings(password=PASSWORD, session_secret="s" * 32, agent_secret="a" * 32,
                           websockify_token="T" * 32, fake_vast_host="127.0.0.1", https_only=False)


@pytest.fixture
def agent():
    agent = AsyncMock()
    agent.health.return_value = AgentHealth(True, True, "sr", False)
    agent.upload.return_value = {"folder": "/workspace/uploads/sr/20260928-120000", "name": "a.mp4", "bytes": 3}
    return agent


@pytest.fixture
def vast():
    return FakeVastClient("127.0.0.1", (8090, 6080))


@pytest.fixture
def client(settings, vast, agent):
    with patch.object(gateway_app, "FAILED_LOGIN_DELAY_SECONDS", 0):
        app = gateway_app.create_app(settings, vast=vast, agent=agent, monitor_in_background=False)
        with TestClient(app) as c:
            yield c


def sign_in(client):
    assert client.post("/api/login", json={"password": PASSWORD}).status_code == 200


class TestAuth:
    @pytest.mark.parametrize("method, path", [
        ("get", "/api/status"), ("post", "/api/start"), ("post", "/api/stop"), ("post", "/api/uploads"),
        ("put", "/api/upload/a.mp4"), ("get", "/desktop/vnc.html"),
    ])
    def test_everything_but_the_page_needs_a_session(self, client, vast, method, path):
        assert getattr(client, method)(path).status_code == 401

    def test_wrong_password_is_refused(self, client):
        assert client.post("/api/login", json={"password": "nope"}).status_code == 401
        assert client.get("/api/me").json()["authenticated"] is False

    def test_sign_in_and_out(self, client):
        sign_in(client)
        assert client.get("/api/me").json()["authenticated"] is True
        client.post("/api/logout")
        assert client.get("/api/status").status_code == 401

    def test_desktop_socket_refuses_without_a_session(self, client):
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect("/desktop/websockify") as ws:
                ws.receive_bytes()
        assert closed.value.code == 1008


class TestStartAndStatus:
    def test_start_launches_the_chosen_language(self, client, vast, agent):
        sign_in(client)
        assert client.post("/api/start", json={"language": "en"}).status_code == 202
        # The start runs as a task on the TestClient's event loop; each request gives it a turn.
        for _ in range(50):
            if agent.session.await_count:
                break
            client.get("/api/status")
        agent.session.assert_awaited_once_with("127.0.0.1:8090", "en")

    def test_unknown_language_is_a_400(self, client):
        sign_in(client)
        assert client.post("/api/start", json={"language": "de"}).status_code == 400

    def test_status_reports_ready_with_the_running_language(self, client, vast):
        sign_in(client)
        vast._status = "running"
        body = client.get("/api/status").json()
        assert (body["state"], body["language"]) == ("ready", "sr")


class TestUpload:
    def test_refused_while_the_appliance_is_off(self, client, agent):
        sign_in(client)
        response = client.put("/api/upload/a.mp4?language=sr&batch=20260928-120000", content=b"abc")
        assert response.status_code == 409
        agent.upload.assert_not_awaited()

    def test_streams_through_to_the_agent(self, client, vast, agent):
        sign_in(client)
        vast._status = "running"
        response = client.put("/api/upload/a%20b.mp4?language=sr&batch=20260928-120000", content=b"abc")
        assert response.status_code == 201
        address, language, batch, filename, _body = agent.upload.await_args.args
        assert (address, language, batch, filename) == ("127.0.0.1:8090", "sr", "20260928-120000", "a b.mp4")

    def test_batches_are_timestamps(self, client):
        sign_in(client)
        batch = client.post("/api/uploads").json()["batch"]
        assert len(batch) == 15 and batch[8] == "-"


class TestProxy:
    def test_the_token_is_added_upstream_and_never_given_to_the_browser(self, client, vast):
        assert websockify_url("1.2.3.4:41002", "T" * 32) == f"ws://1.2.3.4:41002/websockify?token={'T' * 32}"
        page = client.get("/").text
        assert "T" * 32 not in page
