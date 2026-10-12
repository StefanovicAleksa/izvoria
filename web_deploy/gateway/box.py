"""
The rented GPU box: start it when someone signs in, know when it is ready, stop it when nobody needs it.

Adapted from the noVNC demo's controller (deploy/gateway/controller.py), minus the desktop app: the box's services
(the tunnel, llama-server, the API, the ingestion supervisor) start on their own under supervisord, so "ready" simply
means the API's /health says its shared models are resident. The Vast.ai client is reused as it is.

The box holds no data; it reaches this VPS's database and vault through an SSH tunnel it opens itself, which also
forwards its API to 127.0.0.1:BOX_TUNNEL_PORT here (BOX_LINK=tunnel). The box's keys are accepted only while the
gateway has it started: `_set_access` writes "open" before a start and "closed" once it is stopped, and sshd asks
`izvoria-box-keys` (install_gateway.sh), which reads that file, for the keys at every login.
"""

import asyncio
import logging
import ssl
import time
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

import httpx

from deploy.gateway.vast_client import FakeVastClient, InstanceInfo, IVastClient, VastClient, VastError
from web_deploy.gateway.config import GatewaySettings

logger = logging.getLogger(__name__)

POLL_SECONDS = 3.0
CONTAINER_TIMEOUT_SECONDS = 15 * 60
# Loading BGE-M3, the reranker and llama-server's 9B takes a minute or two from a cold page cache
SERVICES_TIMEOUT_SECONDS = 10 * 60
INFO_CACHE_SECONDS = 5.0
HEALTH_CACHE_SECONDS = 2.0


class BoxState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    GPU_UNAVAILABLE = "gpu_unavailable"
    BOOTING = "booting"
    READY = "ready"
    STOPPING = "stopping"
    ERROR = "error"


MESSAGES = {
    BoxState.STOPPED: "The appliance is asleep. It wakes when you sign in.",
    BoxState.STARTING: "Waking the GPU server… this usually takes 1–2 minutes.",
    BoxState.GPU_UNAVAILABLE: ("The GPU is currently rented by someone else. It will start as soon as it is free; "
                               "you can leave this page open."),
    BoxState.BOOTING: "The GPU server is up; loading the language models…",
    BoxState.READY: "Ready.",
    BoxState.STOPPING: "Going to sleep…",
    BoxState.ERROR: "Something went wrong starting the appliance.",
}


@dataclass(frozen=True)
class BoxHealth:
    ready: bool
    busy: bool
    inflight: int = 0
    reports_running: int = 0
    queued: int = 0
    ingesting: int = 0
    starting_error: Optional[str] = None
    # Dev hardware: the live models have handed the GPU to ingestion; chat waits until the queue is empty
    paused: bool = False


@dataclass(frozen=True)
class BoxStatus:
    state: BoxState
    error: str = ""
    health: Optional[BoxHealth] = None

    def as_json(self) -> dict:
        h = self.health
        return {
            "state": self.state.value, "message": MESSAGES[self.state], "error": self.error,
            "queued": h.queued if h else 0, "ingesting": h.ingesting if h else 0,
            "reports_running": h.reports_running if h else 0,
            "paused": h.paused if h else False,
        }


def box_ssl_context(settings: GatewaySettings):
    """
    Pins the box API's own self-signed certificate when it is reached over Vast's public port (BOX_LINK=direct). The
    tunnel needs none: SSH encrypts it, and its end here listens on loopback only.
    """
    if settings.box_cert_file is None or settings.box_link == "tunnel":
        return False
    context = ssl.create_default_context(cafile=str(settings.box_cert_file))
    context.check_hostname = False  # the box is reached by IP; the pinned certificate is what is trusted
    return context


class LocalBox(IVastClient):
    """Dev hardware: the API runs on this machine and is always on. Nothing to start, nothing to stop."""

    def __init__(self, port: int) -> None:
        self._port = port

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def info(self) -> InstanceInfo:
        return InstanceInfo(actual_status="running", intended_status="running", public_ip="127.0.0.1",
                            ports={self._port: self._port})


def build_vast(settings: GatewaySettings) -> IVastClient:
    if settings.box_mode == "local":
        return LocalBox(settings.api_port)
    if settings.fake_box_host:
        return FakeVastClient(settings.fake_box_host, (settings.api_port,))
    return VastClient(settings.vast_api_key, settings.vast_instance_id)


class BoxController:
    def __init__(self, settings: GatewaySettings, vast: IVastClient, http: httpx.AsyncClient,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self._settings = settings
        self._vast = vast
        self._http = http
        self._clock = clock
        self._sleep = sleep
        self._task: Optional[asyncio.Task] = None
        self._step = BoxState.STOPPED
        self._error = ""
        self._scheduling_since: Optional[float] = None
        self._info: Optional[InstanceInfo] = None
        self._info_at = float("-inf")
        self._health: Optional[BoxHealth] = None
        self._health_at = float("-inf")

    @property
    def scheme(self) -> str:
        return "https" if self._settings.box_cert_file and self._settings.box_link == "direct" else "http"

    # ---------- queries ----------

    async def info(self, fresh: bool = False) -> InstanceInfo:
        if fresh or self._info is None or self._clock() - self._info_at > INFO_CACHE_SECONDS:
            self._info = await self._vast.info()
            self._info_at = self._clock()
        return self._info

    async def api_address(self) -> Optional[str]:
        info = await self.info()
        if not info.running:
            return None
        if self._settings.box_link == "tunnel":
            return f"127.0.0.1:{self._settings.box_tunnel_port}"
        return info.address(self._settings.api_port)

    def _set_access(self, open_: bool) -> None:
        path = self._settings.box_access_file
        if path is None:
            return
        value = "open" if open_ else "closed"
        try:
            if path.exists() and path.read_text().strip() == value:
                return
            partial = path.with_name(path.name + ".tmp")
            partial.write_text(value + "\n")
            partial.replace(path)
            logger.info(f"Box SSH access {value}")
        except OSError as exc:
            logger.error(f"Could not set the box's SSH access to {value}: {exc}")

    async def health(self, fresh: bool = False) -> Optional[BoxHealth]:
        if not fresh and self._health is not None and self._clock() - self._health_at < HEALTH_CACHE_SECONDS:
            return self._health
        address = await self.api_address()
        if not address:
            return None
        try:
            response = await self._http.get(f"{self.scheme}://{address}/health", timeout=5.0)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        self._health = BoxHealth(
            ready=bool(body.get("ready")), busy=bool(body.get("busy")), inflight=int(body.get("inflight", 0)),
            reports_running=int(body.get("reports_running", 0)), queued=int(body.get("queued", 0)),
            ingesting=int(body.get("ingesting", 0)), starting_error=body.get("starting_error"),
            paused=bool(body.get("paused")),
        )
        self._health_at = self._clock()
        return self._health

    @property
    def transitioning(self) -> bool:
        return self._task is not None and not self._task.done()

    async def status(self) -> BoxStatus:
        if self.transitioning:
            return BoxStatus(self._step, self._error)
        try:
            info = await self.info()
        except VastError as exc:
            return BoxStatus(BoxState.ERROR, str(exc))
        if not info.running:
            # Stopped from elsewhere (Vast's console, a host failure): its keys go too
            self._set_access(False)
            return BoxStatus(BoxState.STOPPED, self._error)
        health = await self.health()
        if health is None or not health.ready:
            error = health.starting_error if health and health.starting_error else self._error
            return BoxStatus(BoxState.ERROR if health and health.starting_error else BoxState.BOOTING, error or "",
                             health)
        return BoxStatus(BoxState.READY, "", health)

    # ---------- commands ----------

    def start(self) -> None:
        if self.transitioning:
            return
        self._error = ""
        self._step = BoxState.STARTING
        self._task = asyncio.create_task(self._run(self._start()))

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._error = ""
        self._step = BoxState.STOPPING
        self._task = asyncio.create_task(self._run(self._stop()))

    async def wait(self) -> None:
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self, work: Awaitable[None]) -> None:
        try:
            await work
        except asyncio.CancelledError:
            raise
        except (VastError, TimeoutError) as exc:
            logger.warning(f"Box transition failed: {exc}")
            self._error = str(exc)
        except Exception as exc:
            logger.exception("Box transition failed unexpectedly")
            self._error = f"Unexpected error: {exc}"

    async def _start(self) -> None:
        self._scheduling_since = None
        self._set_access(True)  # before the box exists: its tunnel dials in as soon as it boots
        info = await self.info(fresh=True)
        if not info.running:
            await self._vast.start()
            await self._wait_for_container()
        self._step = BoxState.BOOTING
        deadline = self._clock() + SERVICES_TIMEOUT_SECONDS
        while self._clock() < deadline:
            health = await self.health(fresh=True)
            if health is not None and health.starting_error:
                raise VastError(f"The appliance failed to start: {health.starting_error}")
            if health is not None and health.ready:
                return
            await self._sleep(POLL_SECONDS)
        raise TimeoutError("The GPU server started but the appliance didn't come up within 10 minutes.")

    async def _stop(self) -> None:
        # Nothing to shut down gracefully: supervisord stops the API and the ingestion worker re-queues its job
        await self._vast.stop()
        self._set_access(False)
        self._info = None
        self._health = None

    async def _wait_for_container(self) -> None:
        deadline = self._clock() + CONTAINER_TIMEOUT_SECONDS
        while self._clock() < deadline:
            info = await self.info(fresh=True)
            if info.running:
                return
            if info.actual_status == "scheduling":
                self._scheduling_since = self._scheduling_since or self._clock()
                if self._clock() - self._scheduling_since >= self._settings.scheduling_grace_seconds:
                    self._step = BoxState.GPU_UNAVAILABLE
            await self._sleep(POLL_SECONDS)
        raise TimeoutError(MESSAGES[BoxState.GPU_UNAVAILABLE] if self._step is BoxState.GPU_UNAVAILABLE
                           else "The GPU server didn't start within 15 minutes.")
