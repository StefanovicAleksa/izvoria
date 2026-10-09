"""
Forwards a signed-in user's /api request to the GPU box, signed on their behalf, streaming both ways.

Bodies are streamed, never buffered: an upload can be gigabytes and a run's event stream stays open for the length of
a report. Only an allowlist of headers passes in either direction; in particular the browser's cookie never reaches
the box, and nothing the box sends can set one on the firm's origin.
"""

import logging
from typing import AsyncIterator, Optional

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from web_deploy.common.signing import Principal, sign

logger = logging.getLogger(__name__)

_REQUEST_HEADERS = ("content-type", "content-length", "range", "last-event-id", "accept", "if-range")
_RESPONSE_HEADERS = ("content-type", "content-length", "content-range", "accept-ranges", "content-disposition",
                     "cache-control", "x-content-type-options", "content-security-policy", "last-modified", "etag")


async def forward(request: Request, client: httpx.AsyncClient, base_url: str, secret: str,
                  principal: Principal, on_close=None) -> Response:
    path = request.url.path
    query = request.url.query
    headers = {k: v for k, v in request.headers.items() if k.lower() in _REQUEST_HEADERS}
    headers.update(sign(secret, request.method, path, query, principal))
    url = f"{base_url}{path}" + (f"?{query}" if query else "")

    has_body = request.method in ("POST", "PUT", "PATCH")
    upstream_request = client.build_request(
        request.method, url, headers=headers, content=request.stream() if has_body else None,
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0),
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        logger.warning(f"Box unreachable for {request.method} {path}: {exc}")
        if on_close:
            on_close()
        return JSONResponse({"error": "The appliance is not reachable right now.", "kind": "offline"},
                            status_code=503)
    except BaseException:
        # The browser went away mid-upload (ClientDisconnect), or the request was cancelled. Without this the idle
        # monitor counts the request as in flight forever, and the rented box is never stopped.
        if on_close:
            on_close()
        raise

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        except httpx.HTTPError as exc:
            logger.info(f"Upstream stream ended early for {path}: {exc}")

    async def close() -> None:
        await upstream.aclose()
        if on_close:
            on_close()

    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() in _RESPONSE_HEADERS}
    if upstream.headers.get("content-encoding"):
        # aiter_bytes decodes, so the upstream length no longer describes what is sent
        out_headers.pop("content-length", None)
    if out_headers.get("content-type", "").startswith("text/event-stream"):
        out_headers["x-accel-buffering"] = "no"
        out_headers["cache-control"] = "no-cache"
    return StreamingResponse(body(), status_code=upstream.status_code, headers=out_headers,
                             background=BackgroundTask(close))
