from unittest.mock import AsyncMock, patch

import httpx

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from deploy.gateway import app as gateway_app
from deploy.gateway.agent_client import AgentHealth, AgentUnavailable
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
        ("put", "/api/upload/a.mp4"), ("get", "/desktop/vnc.html"), ("get", "/api/sources/stream"),
        ("get", "/api/sources/" + "a" * 22 + "/media"),
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


OFFER = "A" * 21 + "_"


def agent_file(status=206, body=b"0123456789", headers=None):
    """A streamed agent response, as AgentClient.open_source returns it."""
    headers = headers or {
        "content-type": "audio/mp4", "content-length": str(len(body)), "content-range": "bytes 0-9/1000",
        "accept-ranges": "bytes", "content-disposition": "inline", "content-security-policy": "sandbox",
        "x-content-type-options": "nosniff", "cache-control": "private, no-store",
        "server": "uvicorn", "x-agent-internal": "secret-ish",
    }
    return httpx.Response(status, headers=headers, stream=httpx.ByteStream(body))


class TestAppliancePage:
    def test_signed_out_goes_to_the_landing_page(self, client):
        response = client.get("/appliance", follow_redirects=False)
        assert response.status_code == 303 and response.headers["location"] == "/"

    def test_wraps_the_desktop_with_the_sources_panel(self, client):
        sign_in(client)
        page = client.get("/appliance").text
        assert 'src="/desktop/vnc.html?autoconnect=1' in page and '/assets/sources.js' in page
        assert client.get("/assets/sources.js").headers["content-type"].startswith("text/javascript")


class TestSourceFile:
    def test_streams_a_range_through_with_only_the_allowed_headers(self, client, vast, agent):
        sign_in(client)
        vast._status = "running"
        agent.open_source.return_value = agent_file()
        response = client.get(f"/api/sources/{OFFER}/excerpt", headers={"Range": "bytes=0-9"})

        agent.open_source.assert_awaited_once_with("127.0.0.1:8090", OFFER, "excerpt", "bytes=0-9")
        assert response.status_code == 206
        assert response.content == b"0123456789"
        assert response.headers["content-range"] == "bytes 0-9/1000"
        assert response.headers["content-security-policy"] == "sandbox"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "x-agent-internal" not in response.headers

    def test_the_agents_refusal_reaches_the_browser(self, client, vast, agent):
        sign_in(client)
        vast._status = "running"
        agent.open_source.return_value = agent_file(
            404, b'{"error": "No such source."}', {"content-type": "application/json"})
        response = client.get(f"/api/sources/{OFFER}/media")
        assert response.status_code == 404 and response.json() == {"error": "No such source."}

    @pytest.mark.parametrize("offer_id, variant", [
        ("short", "media"), (OFFER, "path"), ("A" * 21 + ".", "media"), ("A" * 23, "original"),
    ])
    def test_only_an_offer_id_and_a_known_variant_get_through(self, client, vast, agent, offer_id, variant):
        sign_in(client)
        vast._status = "running"
        assert client.get(f"/api/sources/{offer_id}/{variant}").status_code == 400
        agent.open_source.assert_not_awaited()

    def test_an_unreachable_box_is_a_bad_gateway(self, client, vast, agent):
        sign_in(client)
        vast._status = "running"
        agent.open_source.side_effect = AgentUnavailable("down")
        assert client.get(f"/api/sources/{OFFER}/media").status_code == 502

    def test_a_playback_counts_as_activity_while_it_streams(self, client, vast, agent):
        sign_in(client)
        vast._status = "running"
        monitor = client.app.state.monitor
        seen = []

        async def body():
            seen.append(monitor._uploads)
            yield b"abc"

        agent.open_source.return_value = httpx.Response(200, headers={"content-type": "video/mp4"},
                                                        stream=_AsyncStream(body()))
        assert client.get(f"/api/sources/{OFFER}/media").content == b"abc"
        assert seen == [1] and monitor._uploads == 0


class _AsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        async for chunk in self._chunks:
            yield chunk
