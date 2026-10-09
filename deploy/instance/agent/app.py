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
from typing import Callable, Collection, Optional
from urllib.parse import quote

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from deploy.instance.agent.activity import ActivitySampler
from deploy.instance.agent.config import AgentSettings
from deploy.instance.agent.media import MediaError, MediaPreparer
from deploy.instance.agent.session import AppSession, UnknownLanguage
from deploy.instance.agent.sources import (
    Offer, OfferRequest, OfferStore, SourceKind, SourceRejected, Variant, resolve_vault_path,
)
from deploy.instance.agent.uploads import UploadRejected, upload_path

logger = logging.getLogger(__name__)

SECRET_HEADER = "x-agent-secret"
OFFER_TOKEN_HEADER = "x-offer-token"
SAMPLE_INTERVAL_SECONDS = 10.0
MAX_WAIT_SECONDS = 30.0
# The desktop app reaches the agent from inside the box. A request through Vast's mapped port never comes from here.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})


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
    offers: Optional[OfferStore] = None,
    preparer: Optional[MediaPreparer] = None,
    loopback_hosts: Collection[str] = LOOPBACK_HOSTS,
) -> Starlette:
    settings = settings or AgentSettings.from_env()
    session = session or AppSession(settings)
    sampler = sampler or ActivitySampler(pid_source=lambda: (session.current() or _NoApp).pid)
    offers = offers or OfferStore(settings.vault)
    preparer = preparer or MediaPreparer(settings.cache_root)

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

    async def offer_source(request: Request) -> JSONResponse:
        """
        The desktop app's "Open source" on the hosted box. Authorised by its per-launch token, not AGENT_SECRET, and
        only from inside the box: this is the one route the app may call, and it can't read anything back.
        """
        if request.client is None or request.client.host not in loopback_hosts:
            return JSONResponse({"error": "forbidden"}, status_code=403)
        token = await asyncio.to_thread(session.offer_token)
        if not token or not hmac.compare_digest(request.headers.get(OFFER_TOKEN_HEADER, ""), token):
            return JSONResponse({"error": "unauthorised"}, status_code=401)
        try:
            offer = await offers.add(OfferRequest.from_json(await request.json()))
        except ValueError as exc:  # SourceRejected, or a body that isn't JSON
            return JSONResponse({"error": str(exc) or "Expected a JSON object."}, status_code=400)
        logger.info(f"Offered {offer.kind.value} source {offer.id} ({offer.request.doc_label}).")
        return JSONResponse({"id": offer.id, "seq": offer.seq}, status_code=201)

    @guard
    async def list_offers(request: Request) -> JSONResponse:
        try:
            after = max(0, int(request.query_params.get("after", "0")))
            wait = min(MAX_WAIT_SECONDS, max(0.0, float(request.query_params.get("wait", "0"))))
        except ValueError:
            return JSONResponse({"error": "'after' and 'wait' must be numbers."}, status_code=400)
        found = await offers.wait_after(after, wait) if wait else offers.after(after)
        return JSONResponse({"latest_seq": offers.latest_seq, "offers": [o.public() for o in found]})

    @guard
    async def serve_offer(request: Request) -> Response:
        offer = offers.get(request.path_params["offer_id"])
        try:
            variant = Variant(request.path_params["variant"])
        except ValueError:
            variant = None
        if offer is None or variant is None or variant not in offer.variants:
            return JSONResponse({"error": "No such source."}, status_code=404)
        # Checked again at serve time: the vault file may have gone, or been swapped for a link, since the offer.
        try:
            resolve_vault_path(str(offer.path), settings.vault)
        except SourceRejected as exc:
            return JSONResponse({"error": str(exc)}, status_code=410)
        try:
            prepared = await preparer.prepare(offer, variant)
        except MediaError as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return FileResponse(prepared.path, media_type=prepared.mime, headers=file_headers(offer, variant, prepared.mime))

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
            Route("/offers", offer_source, methods=["POST"]),
            Route("/offers", list_offers, methods=["GET"]),
            Route("/offers/{offer_id}/{variant}", serve_offer, methods=["GET"]),
        ],
        lifespan=lifespan,
    )


def file_headers(offer: Offer, variant: Variant, mime: str) -> dict:
    """
    Every served file: its type from the agent's own table and never sniffed, no caching, and a sandbox so nothing
    in a client's file can script the gateway's origin. A PDF goes without the sandbox, because Chrome refuses to
    show a PDF under one; its viewer runs no document script in the page.
    """
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"}
    if not (mime == "application/pdf" and variant is Variant.MEDIA):
        headers["Content-Security-Policy"] = "sandbox"
    if variant is Variant.ORIGINAL or offer.kind is SourceKind.DOCUMENT:
        name = offer.public()["download_name"]
        fallback = name.encode("ascii", "replace").decode().replace("?", "_").replace('"', "_")
        headers["Content-Disposition"] = f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(name)}"
    else:
        headers["Content-Disposition"] = "inline"
    return headers


class _NoApp:
    pid = None


async def _sample_forever(sampler: ActivitySampler) -> None:
    while True:
        try:
            await asyncio.to_thread(sampler.sample)
        except Exception as exc:  # a failed sample must not end sampling for the life of the box
            logger.warning(f"Activity sample failed: {exc}")
        await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)
