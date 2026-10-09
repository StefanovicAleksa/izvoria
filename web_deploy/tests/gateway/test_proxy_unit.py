import asyncio

import httpx
import pytest
from starlette.requests import ClientDisconnect, Request

from web_deploy.common.signing import Principal
from web_deploy.gateway.proxy import forward

PRINCIPAL = Principal(firm="legal-demo", user_id="u1", role="member")


def _request(method="PUT", path="/api/uploads/b/file.pdf"):
    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(b"content-type", b"application/pdf")], "server": ("gw", 443), "scheme": "https"}

    async def receive():
        return {"type": "http.request", "body": b"part", "more_body": False}

    return Request(scope, receive)


class Closes:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1


class TestForward:
    """Every forwarded request is counted in flight by the idle monitor until `on_close`; it must always come."""

    @pytest.mark.asyncio
    async def test_a_browser_that_disconnects_mid_upload_is_no_longer_in_flight(self):
        def upstream(request: httpx.Request) -> httpx.Response:
            raise ClientDisconnect()

        closes = Closes()
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            with pytest.raises(ClientDisconnect):
                await forward(_request(), client, "https://box", "s" * 40, PRINCIPAL, on_close=closes)

        assert closes.count == 1

    @pytest.mark.asyncio
    async def test_a_cancelled_request_is_no_longer_in_flight(self):
        def upstream(request: httpx.Request) -> httpx.Response:
            raise asyncio.CancelledError()

        closes = Closes()
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            with pytest.raises(asyncio.CancelledError):
                await forward(_request(), client, "https://box", "s" * 40, PRINCIPAL, on_close=closes)

        assert closes.count == 1

    @pytest.mark.asyncio
    async def test_an_unreachable_box_answers_503_and_is_no_longer_in_flight(self):
        def upstream(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        closes = Closes()
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            response = await forward(_request(), client, "https://box", "s" * 40, PRINCIPAL, on_close=closes)

        assert response.status_code == 503
        assert closes.count == 1

    @pytest.mark.asyncio
    async def test_a_forwarded_response_closes_once_its_body_is_sent(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="ok"))) as client:
            closes = Closes()
            response = await forward(_request("GET", "/api/runs"), client, "https://box", "s" * 40, PRINCIPAL,
                                     on_close=closes)
            assert closes.count == 0

            await response.background()

        assert closes.count == 1
