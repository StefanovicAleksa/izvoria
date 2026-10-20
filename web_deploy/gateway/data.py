"""
The firm data the gateway holds itself: uploads in, cited sources out, and the upload folders cleaned up.

Every firm's vault, uploads and databases live on this VPS; the rented GPU box only computes (docs/web_deploy.md,
"Where data lives"). So a file a user uploads is written here and queued here, without waking anything, and a
source a user opens is cut and served from the vault here. The box is asked only which file and which span a cited
chunk points at (its `/api/sources/{chunk}/locate`), and the answer is checked against the firm's own vault.
"""

import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from deploy.instance.agent.media import MediaError, MediaPreparer
from app.features.storage.utils.upload_record import UploadRecord, read_all, write_record
from web_deploy.common.chunked import ChunkedUpload, discard_incomplete
from web_deploy.common.signing import Principal, sign
from web_deploy.common.source_removal import Connect, existing_sources, postgres_connector, remove_sources, unlink_inside
from web_deploy.common.sources import SourceRejected, Variant, describe, file_headers, offer_for_chunk
from web_deploy.common.tenancy import firm_root
from web_deploy.common.uploads import (
    BatchRejected, UploadJanitor, UploadRejected, batch_dir, batch_files, new_batch_id, receive, sanitize_filename,
)
from web_deploy.gateway.prepare import PrepareSettings, VideoPreparer
from web_deploy.ingest_worker.queue import IngestMode, IngestQueue, QueueError

logger = logging.getLogger(__name__)

JANITOR_INTERVAL_SECONDS = 600.0
PREPARE_POLL_SECONDS = 5.0
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataSettings:
    # The same absolute path as the box's scratch: the database stores absolute file paths
    tenants_root: Path = Path("/workspace/tenants")
    source_cache_root: Path = Path("/workspace/source-cache")
    # The ingestion queue (onyx_control), in this VPS's Postgres
    control_database_url: str = ""
    max_upload_bytes: int = 20 * 1024 ** 3
    # What the Ingest page says about YouTube links (the box resolves and downloads them; LINK_* there too)
    link_ingest_enabled: bool = True
    link_max_videos: int = 200
    # The answer languages a batch may be ingested in: the served language packs (data/languages/*.toml)
    languages: tuple = ()
    # Video batches get their 720p copies made here before the box sees them (prepare.py)
    prepare: PrepareSettings = PrepareSettings()

    @classmethod
    def from_env(cls) -> "DataSettings":
        user = os.environ.get("POSTGRES_USER", "onyx")
        password = os.environ.get("POSTGRES_PASSWORD", "")
        server = os.environ.get("POSTGRES_SERVER", "127.0.0.1")
        port = os.environ.get("POSTGRES_PORT", "5432")
        listed = os.environ.get("INGEST_LANGUAGES", "")
        return cls(
            tenants_root=Path(os.environ.get("TENANTS_ROOT", "/workspace/tenants")),
            source_cache_root=Path(os.environ.get("SOURCE_CACHE_ROOT", "/workspace/source-cache")),
            control_database_url=os.environ.get(
                "CONTROL_DATABASE_URL", f"postgresql+psycopg2://{user}:{password}@{server}:{port}/onyx_control"),
            max_upload_bytes=int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 ** 3))),
            link_ingest_enabled=os.environ.get("LINK_INGEST_ENABLED", "true").strip().lower() in ("1", "true", "yes"),
            link_max_videos=int(os.environ.get("LINK_MAX_VIDEOS", "200")),
            languages=tuple(c.strip() for c in listed.split(",") if c.strip()) or served_languages(),
            prepare=PrepareSettings.from_env(),
        )


def served_languages(root: Path = _REPO_ROOT) -> tuple:
    """The codes of data/languages/*.toml: what the appliance serves (legacy packs live in a subfolder)."""
    return tuple(sorted(p.stem for p in (root / "data" / "languages").glob("*.toml"))) or ("en",)


class DataRoutes:
    """
    The handlers, each taking the signed-in user's Principal. `locate_base` returns the box API's base URL once it is
    ready, or a Response to send instead (the box is waking).
    """

    def __init__(self, settings: DataSettings, queue: IngestQueue, http: httpx.AsyncClient, api_secret: str,
                 locate_base: Callable[[], Awaitable[object]], on_enqueue: Callable[[], Awaitable[None]],
                 preparer: Optional[MediaPreparer] = None, videos: Optional[VideoPreparer] = None,
                 connect_firm: Optional[Connect] = None) -> None:
        self._settings = settings
        self._queue = queue
        self._http = http
        self._secret = api_secret
        self._locate_base = locate_base
        self._on_enqueue = on_enqueue
        self._preparer = preparer or MediaPreparer(settings.source_cache_root)
        self._videos = videos or VideoPreparer(queue, settings.prepare)
        self._connect_firm = connect_firm or postgres_connector()

    # ---------- uploads & the wait list ----------

    async def new_batch(self, request: Request, principal: Principal) -> Response:
        return JSONResponse({"batch": new_batch_id()}, status_code=201)

    async def upload(self, request: Request, principal: Principal) -> Response:
        try:
            folder = self._batch(principal, request.path_params["batch"])
            result = await receive(request.stream(), folder, request.path_params["filename"],
                                   self._settings.max_upload_bytes)
        except (BatchRejected, UploadRejected) as exc:
            return _invalid(exc)
        return JSONResponse(result, status_code=201)

    async def chunks(self, request: Request, principal: Principal) -> Response:
        """Which pieces of a file are here (resumable upload, common/chunked.py); readies it for the rest."""
        data = await _json(request)
        if data is None:
            return _invalid("Expected a JSON object.")
        try:
            upload = self._chunked(principal, request, data.get("size"), data.get("chunk"))
            return JSONResponse(await asyncio.to_thread(upload.begin))
        except (BatchRejected, UploadRejected) as exc:
            return _invalid(exc)

    async def chunk(self, request: Request, principal: Principal) -> Response:
        try:
            index = int(request.path_params["index"])
            upload = self._chunked(principal, request, _int(request.query_params.get("size")),
                                   _int(request.query_params.get("chunk")))
            return JSONResponse(await upload.write(index, request.stream()))
        except (BatchRejected, UploadRejected, ValueError) as exc:
            return _invalid(exc)

    async def record(self, request: Request, principal: Principal) -> Response:
        """
        What files the firm's uploader shrank before sending were (their original hash and name), so ingestion
        dedups and names them as their originals (app/features/storage/utils/upload_record.py).
        """
        data = await _json(request)
        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, dict) or not files:
            return _invalid("Expected {'files': {name: {sha256, original_name}}}.")
        try:
            folder = self._batch(principal, request.path_params["batch"])
            records = {}
            for name, entry in list(files.items())[:10_000]:
                entry = entry if isinstance(entry, dict) else {}
                record = UploadRecord(sha256=str(entry.get("sha256") or "").lower(),
                                      original_name=sanitize_filename(str(entry.get("original_name") or "")))
                if not _SHA256.match(record.sha256):
                    raise BatchRejected(f"'{name}' has no valid sha256.")
                records[sanitize_filename(str(name))] = record
        except (BatchRejected, UploadRejected) as exc:
            return _invalid(exc)
        folder.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(write_record, folder, {**read_all(folder), **records})
        return JSONResponse({"recorded": len(records)})

    def _chunked(self, principal: Principal, request: Request, size, chunk) -> ChunkedUpload:
        folder = self._batch(principal, request.path_params["batch"])
        return ChunkedUpload(folder, request.path_params["filename"], size, chunk, self._settings.max_upload_bytes)

    async def enqueue(self, request: Request, principal: Principal) -> Response:
        try:
            data = await request.json()
        except Exception:
            data = None
        if not isinstance(data, dict):
            return _invalid("Expected a JSON object.")
        try:
            folder = self._batch(principal, str(data.get("batch") or ""))
            mode = IngestMode(str(data.get("mode") or ""))
        except BatchRejected as exc:
            return _invalid(exc)
        except ValueError:
            return _invalid("'mode' must be video, audio, document or image.")
        language = str(data.get("language") or "")
        if language not in self._settings.languages:
            return _invalid(f"Choose a language for the answer: {', '.join(self._settings.languages)}.")
        await asyncio.to_thread(discard_incomplete, folder)  # half-sent files never reach the box
        files = await asyncio.to_thread(batch_files, folder)
        # A video batch waits here for its 720p copies (prepare.py); the box is woken once they are made
        prepare = self._settings.prepare.enabled and mode is IngestMode.VIDEO
        try:
            job = await asyncio.to_thread(self._queue.enqueue, principal.firm, principal.user_id, mode, language,
                                          str(folder), len(files), prepare=prepare)
        except QueueError as exc:
            return _invalid(exc)
        positions = await asyncio.to_thread(self._queue.queue_positions)
        if not prepare:
            await self._on_enqueue()  # the box has work now
        return JSONResponse(job.public(positions.get(job.id)), status_code=201)

    async def jobs(self, request: Request, principal: Principal) -> Response:
        """The firm's wait list. Answered here, so watching a batch being prepared doesn't wake the box."""
        return JSONResponse({"jobs": await asyncio.to_thread(self._queue.list_for_firm, principal.firm),
                             "links": {"enabled": self._settings.link_ingest_enabled,
                                       "max_videos": self._settings.link_max_videos}})

    async def cancel(self, request: Request, principal: Principal) -> Response:
        job_id = request.path_params["job_id"]
        job = await asyncio.to_thread(self._queue.get, job_id)
        if job is None or job.firm != principal.firm:
            return _not_found()
        if job.owner_id != principal.user_id and principal.role not in ("admin", "owner"):
            return JSONResponse({"error": "Only the person who queued it, or a firm admin, can cancel it."},
                                status_code=403)
        if await asyncio.to_thread(self._queue.cancel, job_id, principal.firm):
            return JSONResponse({"cancelled": True})
        # Already running: the box's supervisor ends it within seconds, keeping the files it had finished
        if await asyncio.to_thread(self._queue.request_stop, job_id, principal.firm):
            return JSONResponse({"stopping": True})
        return JSONResponse({"error": "It has already finished.", "kind": "busy"}, status_code=409)

    async def dismiss(self, request: Request, principal: Principal) -> Response:
        """Removes a finished batch from the firm's list (its ✕). Anyone in the firm may: it only tidies the view."""
        job_id = request.path_params["job_id"]
        job = await asyncio.to_thread(self._queue.get, job_id)
        if job is None or job.firm != principal.firm:
            return _not_found()
        if not await asyncio.to_thread(self._queue.dismiss, job_id, principal.firm):
            return JSONResponse({"error": "Only a finished batch can be removed from the list.", "kind": "busy"},
                                status_code=409)
        return JSONResponse({"dismissed": True})

    # ---------- the Library ----------

    async def delete_source(self, request: Request, principal: Principal) -> Response:
        """
        Deletes an ingested file and everything made from it (transcript, index, frames, clips, its vault files), for
        the whole firm. Firm administrators only. Done here, where the firm's database and vault are, so it works while
        the GPU box is stopped.
        """
        if principal.role not in ("admin", "owner"):
            return JSONResponse({"error": "Only a firm administrator can delete files from the Library."},
                                status_code=403)
        try:
            source_id = str(uuid.UUID(request.path_params["source_id"]))
        except ValueError:
            return _not_found()
        if await asyncio.to_thread(self._queue.has_running, principal.firm):
            return JSONResponse({"error": "A batch of yours is being ingested. Delete files once it has finished, "
                                          "or stop it first.", "kind": "busy"}, status_code=409)
        paths = await asyncio.to_thread(self._remove_source, principal.firm, source_id)
        if paths is None:
            return _not_found()
        vault = firm_root(self._settings.tenants_root, principal.firm) / "vault"
        removed = await asyncio.to_thread(unlink_inside, paths, vault)
        logger.info(f"{principal.firm}: source {source_id} deleted by {principal.user_id} ({removed} vault file(s))")
        return JSONResponse({"deleted": True, "files_removed": removed})

    def _remove_source(self, firm: str, source_id: str) -> Optional[list]:
        connection = self._connect_firm(firm)
        try:
            if not existing_sources(connection, [source_id]):
                return None
            return remove_sources(connection, [source_id])
        finally:
            connection.close()

    def _batch(self, principal: Principal, batch: str) -> Path:
        return batch_dir(self._settings.tenants_root, principal.firm, principal.user_id, batch)

    # ---------- cited sources ----------

    async def source_info(self, request: Request, principal: Principal) -> Response:
        offer = await self._offer(request, principal)
        if isinstance(offer, Response):
            return offer
        return JSONResponse(describe(offer, request.query_params.get("label", ""),
                                     request.query_params.get("location", "")))

    async def source_file(self, request: Request, principal: Principal) -> Response:
        try:
            variant = Variant(request.path_params["variant"])
        except ValueError:
            return _not_found()
        offer = await self._offer(request, principal)
        if isinstance(offer, Response):
            return offer
        if variant not in offer.variants:
            return _not_found()
        try:
            prepared = await self._preparer.prepare(offer, variant)
        except MediaError as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        name = describe(offer, request.query_params.get("label", ""),
                        request.query_params.get("location", ""))["download_name"]
        return FileResponse(prepared.path, media_type=prepared.mime,
                            headers=file_headers(offer, variant, prepared.mime, name))

    async def _offer(self, request: Request, principal: Principal):
        base = await self._locate_base()
        if isinstance(base, Response):
            return base
        path = f"/api/sources/{request.path_params['chunk_id']}/locate"
        try:
            response = await self._http.get(f"{base}{path}", headers=sign(self._secret, "GET", path, "", principal),
                                            timeout=30.0)
        except httpx.HTTPError as exc:
            logger.warning(f"Box unreachable locating a source: {exc}")
            return JSONResponse({"error": "The appliance is not reachable right now.", "kind": "offline"},
                                status_code=503)
        if response.status_code != 200:
            return Response(response.content, status_code=response.status_code,
                            media_type=response.headers.get("content-type", "application/json"))
        try:
            location = response.json()
            return offer_for_chunk(location, firm_root(self._settings.tenants_root, principal.firm) / "vault")
        except (ValueError, SourceRejected) as exc:
            return JSONResponse({"error": f"Not found: {exc}", "kind": "not_found"}, status_code=404)

    # ---------- housekeeping ----------

    async def run_video_preparer(self) -> None:
        """Makes the 720p copies of one waiting video batch at a time, then wakes the box for it."""
        if not self._settings.prepare.enabled:
            return
        while True:
            try:
                prepared = await asyncio.to_thread(self._videos.run_once)
            except Exception:
                logger.exception("Preparing videos failed")
                prepared = None
            if prepared is not None:
                logger.info(f"Job {prepared.id}'s videos are prepared; it is in the box's line")
                await self._on_enqueue()
                continue
            await asyncio.sleep(PREPARE_POLL_SECONDS)

    async def run_janitor(self, clock: Callable[[], float] = time.time) -> None:
        janitor = UploadJanitor(self._queue, self._settings.tenants_root)
        while True:
            try:
                await asyncio.to_thread(janitor.run_once, clock())
            except Exception:
                logger.exception("Upload cleanup failed")
            await asyncio.sleep(JANITOR_INTERVAL_SECONDS)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


async def _json(request: Request):
    try:
        data = await request.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _invalid(error) -> JSONResponse:
    return JSONResponse({"error": str(error), "kind": "invalid"}, status_code=400)


def _not_found() -> JSONResponse:
    return JSONResponse({"error": "Not found.", "kind": "not_found"}, status_code=404)

