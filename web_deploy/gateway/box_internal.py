"""
What the GPU box asks of the gateway, through its own tunnel (`-L 127.0.0.1:18000` on the box to this gateway):

- `POST /internal/box/credentials`: the database password the box's role has this time it is started.
- `GET  /internal/box/staging-key`: the VPS's staging public key, which the box installs so the VPS can reach its files.
- `POST /internal/box/stage-in/{job}` and `/stage-out/{job}`: move one running job's files (staging_server.py).

Every route answers only a loopback caller (the tunnel arrives as one; Caddy also refuses /internal/* from outside)
with a service signature (WEB_API_SECRET), and only while the gateway has the box started (the access file says
"open"): a stopped box, or its secrets copied elsewhere, get nothing.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from web_deploy.common.signing import SignatureError, verify_service
from web_deploy.gateway.staging_server import StagingFailed, StagingRefused, StagingServer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoxInternalSettings:
    secret: str
    access_file: Path
    password_file: Path
    staging_public_key: Optional[Path] = None


def box_internal_routes(settings: BoxInternalSettings, staging: Optional[StagingServer],
                        clock: Optional[Callable[[], float]] = None) -> List[Route]:

    def refused(request: Request) -> Optional[Response]:
        if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        try:
            verify_service(settings.secret, request.method, request.url.path, request.headers,
                           now=clock() if clock else None)
        except SignatureError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        if _read(settings.access_file) != "open":
            return JSONResponse({"error": "The box is not started."}, status_code=403)
        return None

    async def credentials(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        password = _read(settings.password_file)
        if not password:
            # Being rotated right now (izvoria-box-access runs as the access file changes): ask again shortly
            return JSONResponse({"error": "Not ready yet."}, status_code=503)
        return JSONResponse({"password": password}, headers={"Cache-Control": "no-store"})

    async def staging_key(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        key = _read(settings.staging_public_key) if settings.staging_public_key else ""
        if not key:
            return JSONResponse({"error": "No staging key on this gateway."}, status_code=503)
        return JSONResponse({"key": key})

    def transfer(action: str):
        async def handler(request: Request) -> Response:
            if (refusal := refused(request)) is not None:
                return refusal
            if staging is None:
                return JSONResponse({"error": "Staging is not set up on this gateway."}, status_code=503)
            job_id = request.path_params["job_id"]
            try:
                result = await asyncio.to_thread(getattr(staging, action), job_id)
            except StagingRefused as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            except StagingFailed as exc:
                logger.error(f"{action} for job {job_id} failed: {exc}")
                return JSONResponse({"error": str(exc)}, status_code=502)
            return JSONResponse(result)
        return handler

    return [
        Route("/internal/box/credentials", credentials, methods=["POST"]),
        Route("/internal/box/staging-key", staging_key, methods=["GET"]),
        Route("/internal/box/stage-in/{job_id}", transfer("stage_in"), methods=["POST"]),
        Route("/internal/box/stage-out/{job_id}", transfer("stage_out"), methods=["POST"]),
    ]


def _read(path: Optional[Path]) -> str:
    try:
        return path.read_text().strip() if path else ""
    except OSError:
        return ""
