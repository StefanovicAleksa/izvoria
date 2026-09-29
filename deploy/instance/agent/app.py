"""
The GPU box's control surface, called only by the gateway (shared secret in X-Agent-Secret).

    ONYX_ROOT=/workspace/onyx_v3 AGENT_SECRET=... /opt/onyx-agent/bin/uvicorn \
        deploy.instance.agent.app:create_app --factory --host 0.0.0.0 --port 8090
"""

import asyncio
import contextlib
import hmac
import logging
import os
import socket
from pathlib import Path
from typing import Callable, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from deploy.instance.agent.activity import ActivitySampler
from deploy.instance.agent.config import AgentSettings
from deploy.instance.agent.session import AppSession, UnknownLanguage
from deploy.instance.agent.uploads import UploadRejected, upload_path

logger = logging.getLogger(__name__)

SECRET_HEADER = "x-agent-secret"
SAMPLE_INTERVAL_SECONDS = 10.0


def postgres_ready() -> bool:
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3):
            return True
    except OSError:
        return False


def display_ready(display: str) -> bool:
    return Path(f"/tmp/.X11-unix/X{display.lstrip(':').split('.')[0]}").exists()


def create_app(
    settings: Optional[AgentSettings] = None,
    session: Optional[AppSession] = None,
    sampler: Optional[ActivitySampler] = None,
    postgres_check: Callable[[], bool] = postgres_ready,
    display_check: Callable[[str], bool] = display_ready,
    sample_in_background: bool = True,
) -> Starlette:
    settings = settings or AgentSettings.from_env()
    session = session or AppSession(settings)
    sampler = sampler or ActivitySampler(pid_source=lambda: (session.current() or _NoApp).pid)

    def authorised(request: Request) -> bool:
        return hmac.compare_digest(request.headers.get(SECRET_HEADER, ""), settings.secret)

    def guard(handler):
        async def wrapped(request: Request):
            if not authorised(request):
                return JSONResponse({"error": "unauthorised"}, status_code=401)
            return await handler(request)
        return wrapped

    @guard
    async def health(request: Request) -> JSONResponse:
        app = session.current()
        return JSONResponse({
            "postgres": await asyncio.to_thread(postgres_check),
            "display": display_check(settings.display),
            "app": {"pid": app.pid, "language": app.language} if app else None,
            "busy": sampler.busy,
        })

    @guard
    async def start_session(request: Request) -> JSONResponse:
        body = await request.json()
        language = str(body.get("language", ""))
        try:
            app = await asyncio.to_thread(session.ensure, language)
        except UnknownLanguage as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"pid": app.pid, "language": app.language})

    @guard
    async def shutdown(request: Request) -> JSONResponse:
        await asyncio.to_thread(session.stop)
        return JSONResponse({"stopped": True})

    @guard
    async def upload(request: Request) -> JSONResponse:
        language = request.path_params["language"]
        if language not in settings.languages:
            return JSONResponse({"error": f"'{language}' is not a language this appliance serves."}, status_code=400)
        try:
            target = upload_path(
                settings.upload_root, language, request.path_params["batch"], request.path_params["filename"]
            )
        except UploadRejected as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and renamed at the end, so a half-sent file never sits in the folder the tester
        # points the Ingest view at.
        partial = target.with_name(target.name + ".part")
        size = 0
        try:
            with partial.open("wb") as handle:
                async for chunk in request.stream():
                    handle.write(chunk)
                    size += len(chunk)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        return JSONResponse({"folder": str(target.parent), "name": target.name, "bytes": size}, status_code=201)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(_sample_forever(sampler)) if sample_in_background else None
        try:
            yield
        finally:
            if task:
                task.cancel()

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/session", start_session, methods=["POST"]),
            Route("/shutdown", shutdown, methods=["POST"]),
            Route("/upload/{language}/{batch}/{filename}", upload, methods=["PUT"]),
        ],
        lifespan=lifespan,
    )


class _NoApp:
    pid = None


async def _sample_forever(sampler: ActivitySampler) -> None:
    while True:
        try:
            await asyncio.to_thread(sampler.sample)
        except Exception as exc:  # a failed sample must not end sampling for the life of the box
            logger.warning(f"Activity sample failed: {exc}")
        await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)
