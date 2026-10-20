"""
Keeps every GPU box on the current version of the app without anyone running anything.

- **The package.** `update_fleet.py` uploads each deployed version to RELEASES_DIR as `<release>.tar.gz` (exactly the
  commit, plus the built web app and its RELEASE stamp) with `<release>.json` (sha256, size, and a digest of the files
  whose change means provision.sh must run again).
- **Boxes fetch it themselves.** At every start, before its services, a box runs `self_update.py`, which asks
  `GET /box-release/{instance}` for the current version, downloads `/package` when it is behind, applies it and
  reports `POST /applied/{release}`. Over the public address with the service signature (WEB_API_SECRET), not the
  tunnel: a box being caught up has no tunnel while another box is in use. The routes hand out the app's code only,
  which every box holds anyway; never client data.
- **Stale boxes are caught up.** `CatchUp` checks every FLEET_CATCHUP_MINUTES for a box behind the current version.
  Only while no box is running: then nobody is using the appliance, no second GPU bills, and the VPS accepts no box's
  tunnel, so the box being updated can't reach any data even while it boots on its old code. It marks it under
  maintenance (the pool then leaves it alone, fleet.py), starts it (its start script updates it and, seeing the
  mark, starts none of its services), waits for its report, and stops it. A box whose GPU another Vast customer is
  using can't start: it is simply tried again next round, costing nothing meanwhile. A box that doesn't ask for its
  version within CONTACT_SECONDS of starting (its start script predates self-updates, or it can't reach the
  gateway), or that fails or times out, is stopped and left alone until the next deploy, so one broken box can't bill
  an hour every round. Such a box needs `update_fleet.py --box <id>` once.
"""

import asyncio
import dataclasses
import json
import logging
import re
import time
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from web_deploy.common.signing import SignatureError, verify_service
from web_deploy.gateway import fleet
from web_deploy.gateway.vast_pool import VastPool, wants_to_run

logger = logging.getLogger(__name__)

RELEASE_NAME = re.compile(r"^[0-9a-f]{7,40}$")
# How long a started box may take to update itself (a provision.sh run reinstalls packages: minutes)
UPDATE_TIMEOUT_SECONDS = 60 * 60
# How long a started box may take to ask for its version (booting the container takes a few minutes)
CONTACT_SECONDS = 15 * 60
POLL_SECONDS = 15.0


@dataclasses.dataclass(frozen=True)
class ReleaseSettings:
    secret: str
    releases_dir: Path
    fleet_file: Path
    instance_ids: Sequence[int]


class UpdateReports:
    """What boxes told the gateway since they were started, kept in this process for CatchUp to see."""

    def __init__(self) -> None:
        self.asked: set = set()  # asked for the current version (their self_update.py runs)
        self.failed: set = set()  # reported a failed update


def package_path(releases_dir: Path, release: str) -> Path:
    return releases_dir / f"{release}.tar.gz"


def meta_path(releases_dir: Path, release: str) -> Path:
    return releases_dir / f"{release}.json"


def release_routes(settings: ReleaseSettings, reports: Optional[UpdateReports] = None,
                   clock: Optional[Callable[[], float]] = None) -> List[Route]:
    reports = reports or UpdateReports()

    def refused(request: Request) -> Optional[Response]:
        try:
            verify_service(settings.secret, request.method, request.url.path, request.headers,
                           now=clock() if clock else None)
        except SignatureError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        if request.path_params["instance"] not in settings.instance_ids:
            return JSONResponse({"error": "Not one of this gateway's boxes."}, status_code=404)
        return None

    def current() -> Optional[dict]:
        release = fleet.load(settings.fleet_file).release
        if not release or not RELEASE_NAME.match(release):
            return None
        try:
            meta = json.loads(meta_path(settings.releases_dir, release).read_text())
        except (OSError, ValueError):
            return None
        return meta if package_path(settings.releases_dir, release).is_file() else None

    async def describe(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        meta = current()
        if meta is None:
            return JSONResponse({"error": "No release package on this gateway."}, status_code=404)
        instance = request.path_params["instance"]
        reports.asked.add(instance)
        return JSONResponse({**meta, "maintenance": instance in fleet.load(settings.fleet_file).maintenance},
                            headers={"Cache-Control": "no-store"})

    async def package(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        meta = current()
        if meta is None:
            return JSONResponse({"error": "No release package on this gateway."}, status_code=404)
        return FileResponse(package_path(settings.releases_dir, meta["release"]), media_type="application/gzip")

    async def applied(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        instance, release = request.path_params["instance"], request.path_params["release"]
        if not RELEASE_NAME.match(release):
            return JSONResponse({"error": "Not a release."}, status_code=400)
        state = fleet.load(settings.fleet_file)
        fleet.save(settings.fleet_file, dataclasses.replace(state, boxes={**state.boxes, instance: release}))
        logger.info(f"GPU box {instance} now holds {release}" +
                    ("" if release == state.release else f" (the current version is {state.release})"))
        return JSONResponse({"recorded": release})

    async def failed(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        instance = request.path_params["instance"]
        reports.failed.add(instance)
        logger.error(f"GPU box {instance} couldn't update itself; see /workspace/logs/self_update.log on it")
        return JSONResponse({"recorded": "failed"})

    return [
        Route("/box-release/{instance:int}", describe, methods=["GET"]),
        Route("/box-release/{instance:int}/failed", failed, methods=["POST"]),
        Route("/box-release/{instance:int}/package", package, methods=["GET"]),
        Route("/box-release/{instance:int}/applied/{release}", applied, methods=["POST"]),
    ]


class CatchUp:
    """Brings boxes that missed a deploy up to the current version, one at a time, while nobody needs them."""

    def __init__(self, pool: VastPool, fleet_file: Path, instance_ids: Sequence[int], interval_seconds: float,
                 reports: Optional[UpdateReports] = None,
                 timeout_seconds: float = UPDATE_TIMEOUT_SECONDS, contact_seconds: float = CONTACT_SECONDS, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self._pool = pool
        self._fleet_file = fleet_file
        self._ids = list(instance_ids)
        self._reports = reports or UpdateReports()
        self._interval = interval_seconds
        self._timeout = timeout_seconds
        self._contact = contact_seconds
        self._given_up: Dict[int, str] = {}  # box -> the version it couldn't be brought to
        self._clock = clock
        self._sleep = sleep

    def stale(self) -> List[int]:
        state = fleet.load(self._fleet_file)
        if not state.release:
            return []
        return [i for i in self._ids if not state.current(i) and i not in state.maintenance
                and self._given_up.get(i) != state.release]

    async def run_forever(self) -> None:
        while True:
            await self._sleep(self._interval)
            try:
                await self.round()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Catching up a stale GPU box failed; trying again next round")

    async def round(self) -> Optional[str]:
        """Tries the stale boxes in turn until one starts, and updates it. Returns what happened, for the log."""
        stale = self.stale()
        if not stale:
            return None
        infos = await self._pool.describe()
        if any(wants_to_run(info) for info in infos.values()):
            return None  # the appliance is in use (or an update is running): wait for a quiet moment
        for instance in stale:
            if instance not in infos:
                continue  # destroyed
            outcome = await self._update(instance)
            logger.info(f"Catch-up of GPU box {instance}: {outcome}")
            if outcome not in ("updated", "GPU taken"):
                self._given_up[instance] = fleet.load(self._fleet_file).release
                logger.error(f"GPU box {instance} is left on its old version until the next deploy; update it with "
                             f"`update_fleet.py --box {instance}`")
            if outcome != "GPU taken":
                return outcome
        return "every stale box's GPU is taken"

    async def _update(self, instance: int) -> str:
        self._mark(instance, True)
        self._reports.failed.discard(instance)
        self._reports.asked.discard(instance)
        try:
            reason = await self._pool.start_one(instance)
            if reason is not None:
                return "GPU taken"
            try:
                started = self._clock()
                deadline = started + self._timeout
                while self._clock() < deadline:
                    if fleet.load(self._fleet_file).current(instance):
                        return "updated"
                    if instance in self._reports.failed:
                        return "failed (see /workspace/logs/self_update.log on the box)"
                    if instance not in self._reports.asked and self._clock() - started > self._contact:
                        return "never asked for its version (its start script can't update itself)"
                    await self._sleep(POLL_SECONDS)
                logger.error(f"GPU box {instance} was started to update itself but didn't report the current "
                             f"version within {int(self._timeout / 60)} min; see /workspace/logs/self_update.log "
                             f"on it")
                return "timed out"
            finally:
                await self._pool.stop_one(instance)
        finally:
            self._mark(instance, False)

    def _mark(self, instance: int, updating: bool) -> None:
        state = fleet.load(self._fleet_file)
        marked = set(state.maintenance)
        if updating:
            marked.add(instance)
        else:
            marked.discard(instance)
        fleet.save(self._fleet_file, dataclasses.replace(state, maintenance=frozenset(marked)))
