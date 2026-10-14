"""
Several rented GPU boxes behind one: the primary and its spares, at most one of them running.

Why: a stopped Vast instance keeps its disk but not its GPU. Anyone may rent that GPU meanwhile, and the instance then
can't start until they let it go, which can take days (it happened on 2026-10-14). Each spare is a fully provisioned box
on another machine, costing only its disk while stopped, so the appliance wakes on whichever one has its GPU free.

    VAST_INSTANCE_ID=55919995,55823057      preference order: the first that can start is used

The controller (box.py) sees one IVastClient, exactly as with a single box:
- start() tries each box in order. One whose GPU is taken (Vast refuses, or leaves it "scheduling" past the grace
  period) is stopped again at once, because Vast queues a refused start and would otherwise run it later, billing a
  second GPU nobody uses; then the next is tried.
- info() describes the box in use. Any other box that is running or about to run is stopped: only one ever bills.
- stop() stops every box.
All boxes hold the same code and models and the same tunnel identity's place on the VPS (each its own key, added with
izvoria-add-box-keys), and none holds data, so it doesn't matter which one answers.
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

import httpx

from deploy.gateway.vast_client import InstanceInfo, IVastClient, VastClient, VastError, _parse_ports

logger = logging.getLogger(__name__)

POLL_SECONDS = 3.0
# A stop just sent takes Vast a few seconds to show; don't send it again on every status poll meanwhile
RESTOP_SECONDS = 60.0
# Vast's states for an instance that has its GPU (or is getting it): from here on it is billed
ALLOCATED = ("running", "loading", "created")

Lister = Callable[[], Awaitable[Dict[int, InstanceInfo]]]


def wants_to_run(info: InstanceInfo) -> bool:
    """Running, booting, or with a start pending (Vast's queued start): it bills, or is about to."""
    return info.actual_status in ALLOCATED or info.intended_status == "running"


class NoGpuAvailable(VastError):
    pass


class VastPool(IVastClient):
    def __init__(self, members: Dict[int, IVastClient], grace_seconds: float = 60.0,
                 lister: Optional[Lister] = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        if not members:
            raise ValueError("VastPool needs at least one instance.")
        self._members = dict(members)
        self._order: List[int] = list(members)
        self._grace = grace_seconds
        self._lister = lister
        self._clock = clock
        self._sleep = sleep
        self._active = self._order[0]
        self._starting = False
        self._stopped_at: Dict[int, float] = {}

    @property
    def active(self) -> int:
        return self._active

    # ---------- IVastClient ----------

    async def info(self) -> InstanceInfo:
        infos = await self._infos()
        if not self._starting:
            await self._keep_one(infos)
        if self._active not in infos:  # destroyed: carry on with the first box that still exists
            self._active = next(i for i in self._order if i in infos)
        return infos[self._active]

    async def start(self) -> None:
        self._starting = True
        try:
            refusals = []
            for candidate in self._order:
                infos = await self._infos()
                if candidate not in infos:
                    refusals.append(f"{candidate}: no longer exists")
                    continue
                for other, info in infos.items():
                    if other != candidate and wants_to_run(info):
                        await self._stop_one(other, "another box is being started")
                self._active = candidate
                reason = await self._try(candidate)
                if reason is None:
                    logger.info(f"GPU box {candidate} is starting")
                    return
                refusals.append(f"{candidate}: {reason}")
                logger.warning(f"GPU box {candidate} can't start ({reason}); trying the next")
                await self._stop_one(candidate, "its GPU is taken", force=True)
            self._active = self._order[0]
            logger.error("No GPU box could start: " + "; ".join(refusals))
            raise NoGpuAvailable("The GPU servers are all in use by other customers of the GPU provider right now. "
                                 "Try again in a while.")
        finally:
            self._starting = False

    async def stop(self) -> None:
        try:
            infos = await self._infos()
            targets = [i for i in self._order if i in infos and wants_to_run(infos[i])]
        except VastError:
            targets = list(self._order)  # can't see them: stop them all rather than leave one billing
        errors = []
        for instance in targets:
            try:
                await self._members[instance].stop()
                self._stopped_at[instance] = self._clock()
            except VastError as exc:
                errors.append(f"{instance}: {exc}")
        if errors:
            raise VastError("Could not stop every GPU box: " + "; ".join(errors))

    # ---------- plumbing ----------

    async def _try(self, instance: int) -> Optional[str]:
        """None once the box has its GPU, else why not (Vast refused, or it waited for the GPU past the grace)."""
        member = self._members[instance]
        try:
            await member.start()
        except VastError as exc:
            return str(exc)
        deadline = self._clock() + self._grace
        while True:
            try:
                info = await member.info()
            except VastError as exc:
                return str(exc)
            if info.actual_status in ALLOCATED:
                return None
            if self._clock() >= deadline:
                return f"still waiting for its GPU after {int(self._grace)} s ({info.actual_status})"
            await self._sleep(POLL_SECONDS)

    async def _keep_one(self, infos: Dict[int, InstanceInfo]) -> None:
        live = [i for i in self._order if i in infos and wants_to_run(infos[i])]
        if not live:
            return
        running = [i for i in live if infos[i].actual_status in ALLOCATED]
        keep = self._active if self._active in (running or live) else (running or live)[0]
        self._active = keep
        for extra in live:
            if extra != keep:
                await self._stop_one(extra, f"box {keep} is the one in use")

    async def _stop_one(self, instance: int, why: str, force: bool = False) -> None:
        if not force and self._clock() - self._stopped_at.get(instance, float("-inf")) < RESTOP_SECONDS:
            return
        logger.warning(f"Stopping GPU box {instance}: {why}")
        self._stopped_at[instance] = self._clock()
        try:
            await self._members[instance].stop()
        except VastError as exc:
            logger.error(f"Could not stop GPU box {instance}: {exc}")

    async def _infos(self) -> Dict[int, InstanceInfo]:
        if self._lister is not None:
            return await self._lister()
        infos = {}
        for instance, member in self._members.items():
            try:
                infos[instance] = await member.info()
            except VastError as exc:
                logger.warning(f"GPU box {instance}: {exc}")
        if not infos:
            raise VastError("Vast reports none of the GPU boxes.")
        return infos


def build_pool(api_key: str, instance_ids: Sequence[int], grace_seconds: float,
               base_url: str = "https://console.vast.ai",
               transport: Optional[httpx.AsyncBaseTransport] = None) -> VastPool:
    """The production pool: one VastClient per box for start/stop, and one listing call for every box's state."""
    members = {i: VastClient(api_key, i, base_url=base_url, transport=transport) for i in instance_ids}
    client = httpx.AsyncClient(headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0, transport=transport)
    url = f"{base_url.rstrip('/')}/api/v1/instances/"  # v0 answers 410 Gone since 2026

    async def lister() -> Dict[int, InstanceInfo]:
        try:
            response = await client.get(url, params={"owner": "me"})
            response.raise_for_status()
            rows = response.json().get("instances") or []
        except (httpx.HTTPError, ValueError) as exc:
            raise VastError(f"Vast API call failed: {exc}") from exc
        infos = {}
        for row in rows:
            try:
                instance = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            if instance in members:
                infos[instance] = InstanceInfo(
                    actual_status=str(row.get("actual_status") or "unknown"),
                    intended_status=str(row.get("intended_status") or "unknown"),
                    public_ip=row.get("public_ipaddr") or None,
                    ports=_parse_ports(row.get("ports") or {}),
                    status_msg=str(row.get("status_msg") or ""),
                )
        if not infos:
            raise VastError("Vast reports none of the GPU boxes; they may have been destroyed.")
        return infos

    return VastPool(members, grace_seconds=grace_seconds, lister=lister)
