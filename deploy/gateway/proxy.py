"""
/desktop/* -> the GPU box's noVNC (websockify serves noVNC's static files and the VNC WebSocket on one port).

The websockify token is added here and never reaches the browser: websockify's TokenFile plugin both routes and
authenticates by it, so the GPU box's public port is useless without it.
"""

import asyncio
import logging
from typing import Callable, ContextManager, Optional

import httpx
from starlette.responses import Response, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

_HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
                "proxy-authorization", "proxy-authenticate", "content-encoding", "content-length"}


# What a cited file's response may carry to the browser. Anything else the agent sends stays on the VPS.
_SOURCE_HEADERS = ("content-type", "content-length", "content-range", "accept-ranges", "content-disposition",
                   "content-security-policy", "x-content-type-options", "cache-control")


def stream_source(upstream: httpx.Response, around: Callable[[], ContextManager]) -> StreamingResponse:
    """
    Relays one offered file, chunk by chunk, so a two-hour recording never sits in this small VPS's memory, with
    the agent's status kept (206 for a Range request, which is what lets the browser seek). `around` is held for as
    long as bytes flow, so the idle monitor counts a playback as use.
    """
    headers = {name: upstream.headers[name] for name in _SOURCE_HEADERS if name in upstream.headers}

    async def body():
        try:
            with around():
                async for chunk in upstream.aiter_raw():
                    yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body(), status_code=upstream.status_code, headers=headers)


def websockify_url(address: str, token: str) -> str:
    return f"ws://{address}/websockify?token={token}"


async def proxy_static(client: httpx.AsyncClient, address: str, path: str, query: str) -> Response:
    url = f"http://{address}/{path}" + (f"?{query}" if query else "")
    try:
        upstream = await client.get(url)
    except httpx.HTTPError:
        return Response("The appliance's desktop isn't reachable yet.", status_code=502)
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_HEADERS}
    return Response(upstream.content, status_code=upstream.status_code, headers=headers)


async def proxy_websocket(browser: WebSocket, address: str, token: str) -> None:
    requested = browser.scope.get("subprotocols") or []
    try:
        upstream = await connect(
            websockify_url(address, token),
            subprotocols=requested or None,
            max_size=None,
            open_timeout=15,
        )
    except Exception as exc:  # any failure to reach the box means the same thing to the browser
        logger.warning(f"Desktop WebSocket upstream failed: {exc}")
        await browser.close(code=1011)
        return

    await browser.accept(subprotocol=upstream.subprotocol)
    async with upstream:
        to_box = asyncio.create_task(_browser_to_box(browser, upstream))
        to_browser = asyncio.create_task(_box_to_browser(upstream, browser))
        done, pending = await asyncio.wait({to_box, to_browser}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
    try:
        await browser.close()
    except (RuntimeError, WebSocketDisconnect):
        pass  # already closed by the browser


async def _browser_to_box(browser: WebSocket, upstream) -> None:
    try:
        while True:
            message = await browser.receive()
            if message["type"] == "websocket.disconnect":
                return
            data: Optional[bytes] = message.get("bytes")
            await upstream.send(data if data is not None else message.get("text", ""))
    except (WebSocketDisconnect, ConnectionClosed):
        return


async def _box_to_browser(upstream, browser: WebSocket) -> None:
    try:
        async for data in upstream:
            if isinstance(data, bytes):
                await browser.send_bytes(data)
            else:
                await browser.send_text(data)
    except (WebSocketDisconnect, ConnectionClosed, RuntimeError):
        return
