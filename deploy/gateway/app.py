"""
The always-on demo gateway. Runs behind Caddy (TLS) on a small VPS:

    /opt/onyx-gateway/bin/uvicorn deploy.gateway.app:create_app --factory --host 127.0.0.1 --port 8000 \
        --proxy-headers --ws-max-size 16777216
"""

import asyncio
import contextlib
import hmac
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from deploy.gateway.agent_client import AgentClient, AgentRefused, AgentUnavailable
from deploy.gateway.config import AGENT_PORT, LANGUAGES, NOVNC_PORT, GatewaySettings
from deploy.gateway.controller import ApplianceController, ApplianceState
from deploy.gateway.idle_monitor import IdleMonitor
from deploy.gateway.proxy import proxy_static, proxy_websocket, stream_source
from deploy.gateway.sources import OFFER_ID, VARIANTS, source_events
from deploy.gateway.vast_client import FakeVastClient, IVastClient, VastClient

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
FAILED_LOGIN_DELAY_SECONDS = 1.0


def create_app(
    settings: Optional[GatewaySettings] = None,
    vast: Optional[IVastClient] = None,
    agent: Optional[AgentClient] = None,
    monitor_in_background: bool = True,
) -> Starlette:
    settings = settings or GatewaySettings.from_env()
    if vast is None:
        vast = (FakeVastClient(settings.fake_vast_host, (AGENT_PORT, NOVNC_PORT)) if settings.fake_vast_host
                else VastClient(settings.vast_api_key, settings.vast_instance_id, settings.vast_api_url))
    agent = agent or AgentClient(settings.agent_secret)
    controller = ApplianceController(settings, vast, agent)
    monitor = IdleMonitor(controller, settings.idle_minutes * 60)
    static_client = httpx.AsyncClient(timeout=15.0)

    def signed_in(scope_holder) -> bool:
        return bool(scope_holder.session.get("auth"))

    def guard(handler):
        async def wrapped(request: Request):
            if not signed_in(request):
                return JSONResponse({"error": "Sign in first."}, status_code=401)
            return await handler(request)
        return wrapped

    async def index(request: Request) -> Response:
        return FileResponse(STATIC_DIR / "index.html")

    async def appliance(request: Request) -> Response:
        # The desktop and its Sources panel. Signed out, the landing page is where to start.
        if not signed_in(request):
            return RedirectResponse("/", status_code=303)
        return FileResponse(STATIC_DIR / "appliance.html", headers={"Cache-Control": "no-store"})

    async def sources_script(request: Request) -> Response:
        return FileResponse(STATIC_DIR / "sources.js", media_type="text/javascript")

    async def me(request: Request) -> JSONResponse:
        return JSONResponse({"authenticated": signed_in(request), "languages": LANGUAGES,
                             "idle_minutes": settings.idle_minutes})

    async def login(request: Request) -> JSONResponse:
        try:
            password = str((await request.json()).get("password", ""))
        except ValueError:
            password = ""
        if not hmac.compare_digest(password.encode(), settings.password.encode()):
            await asyncio.sleep(FAILED_LOGIN_DELAY_SECONDS)
            return JSONResponse({"error": "Wrong password."}, status_code=401)
        request.session["auth"] = True
        return JSONResponse({"authenticated": True})

    async def logout(request: Request) -> JSONResponse:
        request.session.clear()
        return JSONResponse({"authenticated": False})

    @guard
    async def status(request: Request) -> JSONResponse:
        monitor.touch()
        return JSONResponse((await controller.status()).as_json())

    @guard
    async def start(request: Request) -> JSONResponse:
        language = str((await request.json()).get("language", ""))
        try:
            controller.start(language)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        monitor.touch()
        return JSONResponse((await controller.status()).as_json(), status_code=202)

    @guard
    async def stop(request: Request) -> JSONResponse:
        controller.stop()
        return JSONResponse((await controller.status()).as_json(), status_code=202)

    @guard
    async def new_upload_batch(request: Request) -> JSONResponse:
        return JSONResponse({"batch": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")})

    @guard
    async def upload(request: Request) -> JSONResponse:
        language = request.query_params.get("language", "")
        batch = request.query_params.get("batch", "")
        if language not in LANGUAGES:
            return JSONResponse({"error": "Choose Serbian or English first."}, status_code=400)
        view = await controller.status()
        address = await controller.agent_address()
        if view.state not in (ApplianceState.READY, ApplianceState.BUSY) or not address:
            return JSONResponse({"error": "Start the appliance before uploading."}, status_code=409)
        with monitor.transfer_in_flight():
            try:
                result = await agent.upload(address, language, batch, request.path_params["filename"],
                                            request.stream())
            except AgentRefused as exc:
                return JSONResponse({"error": str(exc)}, status_code=exc.status_code)
            except AgentUnavailable as exc:
                return JSONResponse({"error": str(exc)}, status_code=502)
        return JSONResponse(result, status_code=201)

    @guard
    async def sources_stream(request: Request) -> Response:
        """The Sources panel's feed of what the tester opened in the desktop app (Server-Sent Events)."""
        events = source_events(agent, controller.agent_address, request.headers.get("last-event-id"))
        return StreamingResponse(events, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @guard
    async def source_file(request: Request) -> Response:
        """One offered file, streamed from the box. Only an opaque id and a fixed variant name come from the browser."""
        offer_id, variant = request.path_params["offer_id"], request.path_params["variant"]
        if not OFFER_ID.match(offer_id) or variant not in VARIANTS:
            return JSONResponse({"error": "No such source."}, status_code=400)
        address = await controller.agent_address()
        if not address:
            return JSONResponse({"error": "The appliance is off."}, status_code=409)
        try:
            upstream = await agent.open_source(address, offer_id, variant, request.headers.get("range"))
        except AgentUnavailable as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)
        return stream_source(upstream, monitor.transfer_in_flight)

    @guard
    async def desktop_static(request: Request) -> Response:
        address = await controller.desktop_address()
        if not address:
            return Response("The appliance is off. Start it from the home page.", status_code=409)
        return await proxy_static(static_client, address, request.path_params["path"], request.url.query)

    async def desktop_socket(websocket: WebSocket) -> None:
        if not signed_in(websocket):
            await websocket.close(code=1008)
            return
        address = await controller.desktop_address()
        if not address:
            await websocket.close(code=1013)
            return
        with monitor.socket_open():
            await proxy_websocket(websocket, address, settings.websockify_token)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(monitor.run_forever()) if monitor_in_background else None
        try:
            yield
        finally:
            if task:
                task.cancel()
            await static_client.aclose()
            await agent.aclose()

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/appliance", appliance),
            Route("/assets/sources.js", sources_script),
            Route("/api/me", me),
            Route("/api/login", login, methods=["POST"]),
            Route("/api/logout", logout, methods=["POST"]),
            Route("/api/status", status),
            Route("/api/start", start, methods=["POST"]),
            Route("/api/stop", stop, methods=["POST"]),
            Route("/api/uploads", new_upload_batch, methods=["POST"]),
            Route("/api/upload/{filename}", upload, methods=["PUT"]),
            Route("/api/sources/stream", sources_stream),
            Route("/api/sources/{offer_id}/{variant}", source_file),
            WebSocketRoute("/desktop/websockify", desktop_socket),
            Route("/desktop/{path:path}", desktop_static),
        ],
        middleware=[
            # SameSite=strict is the CSRF defence: no other site can make the browser send this cookie.
            Middleware(SessionMiddleware, secret_key=settings.session_secret, session_cookie="onyx_demo",
                       max_age=7 * 24 * 3600, same_site="strict", https_only=settings.https_only),
        ],
        lifespan=lifespan,
    )
    app.state.controller = controller
    app.state.monitor = monitor
    return app
