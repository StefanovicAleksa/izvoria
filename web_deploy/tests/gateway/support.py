"""What the gateway tests share: a fake GPU box API, a mail transport that records, and the two sign-in steps."""

from typing import List

import httpx
from starlette.testclient import TestClient

SECRET = "k" * 40
PASSWORD = "correct horse battery"
FIRM_PASSWORD = "legal demo shared secret"
OTHER_FIRM_PASSWORD = "other firm shared secret"
CONSOLE_PASSWORD = "the owner's console door"
HOST = "onyx.test"


class BoxApi:
    """The GPU box API as the gateway sees it: /health, and records of every forwarded request."""

    def __init__(self):
        self.ready = True
        self.forwarded = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": self.ready, "busy": False})
        self.forwarded.append(request)
        return httpx.Response(202, json={"run_id": "r1"}, headers={"set-cookie": "evil=1", "x-internal": "no"})


class InlineExecutor:
    """Runs mail on the calling thread, so a test sees the outcome as soon as the request returns."""

    def submit(self, fn, *args):
        fn(*args)

    def shutdown(self, wait=True, cancel_futures=False):
        pass


class RecordingTransport:
    def __init__(self, fail: bool = False):
        self.sent: List = []
        self.fail = fail

    def __call__(self, settings, message):
        if self.fail:
            raise ConnectionRefusedError("smtp is down")
        self.sent.append(message)


def client_for(app, host: str = HOST, origin: str = None, address: str = "testclient") -> TestClient:
    return TestClient(app, base_url=f"http://{host}", headers={"origin": origin or f"http://{host}"},
                      client=(address, 50000))


def firm_step(client: TestClient, firm: str = "legal-demo", password: str = FIRM_PASSWORD, remember: bool = False):
    return client.post("/auth/firm", json={"firm": firm, "password": password, "remember": remember})


def signed_in(app, firm: str = "legal-demo", email: str = "alice@firm.com", password: str = PASSWORD,
              firm_password: str = FIRM_PASSWORD) -> TestClient:
    client = client_for(app)
    assert firm_step(client, firm, firm_password).status_code == 200
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return client
