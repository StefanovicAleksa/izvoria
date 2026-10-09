"""Dev hardware: the same two-step sign-in over plain HTTP on a LAN, and a GPU box that is this machine."""

import httpx
import pytest
from starlette.testclient import TestClient

from web_deploy.common.signing import verify
from web_deploy.gateway.app import create_app
from web_deploy.gateway.box import BoxController, LocalBox
from web_deploy.gateway.config import GatewaySettings
from web_deploy.gateway.idle import IdleMonitor
from web_deploy.gateway.store import OWNER_FIRM, Store

SECRET = "k" * 40
PASSWORD = "correct horse battery"
FIRM_PASSWORD = "legal demo shared secret"
OTHER_PASSWORD = "other firm shared secret"
CONSOLE_PASSWORD = "the owner's console door"
LAN = "192.168.1.50:8000"


class BoxApi:
    def __init__(self):
        self.paused = False
        self.forwarded = []

    def __call__(self, request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True, "busy": self.paused, "paused": self.paused,
                                             "ingesting": int(self.paused)})
        self.forwarded.append(request)
        return httpx.Response(200, json={"ok": True})


@pytest.fixture
def box_api():
    return BoxApi()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "g.sqlite3")
    s.create_firm("legal-demo", "Legal demo", password=FIRM_PASSWORD)
    s.create_firm("other-firm", "Other firm", "sr", password=OTHER_PASSWORD)
    s.create_user("legal-demo", "alice@firm.com", PASSWORD)
    s.create_user("other-firm", "alice@firm.com", "a different password")
    s.create_user(OWNER_FIRM, "me@home.lan", PASSWORD, "owner")
    s.set_firm_password(OWNER_FIRM, CONSOLE_PASSWORD)
    return s


@pytest.fixture
def app(tmp_path, store, box_api):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>spa</html>")
    settings = GatewaySettings(api_secret=SECRET, database_path=tmp_path / "g.db", static_dir=static,
                               https_only=False, box_mode="local")
    http = httpx.AsyncClient(transport=httpx.MockTransport(box_api))
    box = BoxController(settings, LocalBox(settings.api_port), http)
    return create_app(settings, store, box, IdleMonitor(box, 1800), http, run_idle_monitor=False)


def lan(app):
    return TestClient(app, base_url=f"http://{LAN}", headers={"origin": f"http://{LAN}"})


def sign_in(client, firm, firm_password, email, password=PASSWORD):
    assert client.post("/auth/firm", json={"firm": firm, "password": firm_password}).status_code == 200
    return client.post("/auth/login", json={"email": email, "password": password})


class TestOnTheLan:
    def test_before_sign_in_the_app_asks_for_the_firm(self, app):
        client = lan(app)
        assert client.get("/").text == "<html>spa</html>"
        assert client.get("/auth/me").json()["stage"] == "firm"

    def test_the_firm_signed_in_to_decides_the_account(self, app):
        client = lan(app)
        assert sign_in(client, "other-firm", OTHER_PASSWORD, "alice@firm.com").status_code == 401
        client.post("/auth/firm/forget")
        assert sign_in(client, "Legal-Demo", FIRM_PASSWORD, "alice@firm.com").status_code == 200
        me = client.get("/auth/me").json()
        assert (me["console"], me["firm"]["slug"], me["user"]["email"]) == ("firm", "legal-demo", "alice@firm.com")

    def test_api_calls_go_to_the_local_box_signed_for_the_users_firm(self, app, store, box_api):
        client = lan(app)
        sign_in(client, "other-firm", OTHER_PASSWORD, "alice@firm.com", "a different password")
        assert client.get("/api/conversations").status_code == 200
        forwarded = box_api.forwarded[-1]
        assert str(forwarded.url).startswith("http://127.0.0.1:8095/api/conversations")
        assert verify(SECRET, "GET", "/api/conversations", "", forwarded.headers).firm == "other-firm"

    def test_the_owner_signs_in_to_the_console_as_firm_admin(self, app):
        client = lan(app)
        assert sign_in(client, "admin", CONSOLE_PASSWORD, "me@home.lan").status_code == 200
        assert client.get("/auth/me").json()["console"] == "admin"
        assert client.get("/admin/api/firms").status_code == 200
        assert client.get("/api/conversations").status_code == 404

    def test_the_cookies_work_over_plain_http_on_the_lan(self, app):
        client = lan(app)
        firm = client.post("/auth/firm", json={"firm": "legal-demo", "password": FIRM_PASSWORD})
        account = client.post("/auth/login", json={"email": "alice@firm.com", "password": PASSWORD})
        assert "Secure" not in firm.headers["set-cookie"] and "Secure" not in account.headers["set-cookie"]


class TestLocalBox:
    def test_always_ready_and_reports_when_ingestion_has_the_gpu(self, app, box_api):
        client = lan(app)
        sign_in(client, "legal-demo", FIRM_PASSWORD, "alice@firm.com")
        assert client.get("/box/status").json()["state"] == "ready"
        box_api.paused = True
        import time
        time.sleep(2.1)  # past the health cache
        status = client.get("/box/status").json()
        assert (status["state"], status["paused"], status["ingesting"]) == ("ready", True, 1)
