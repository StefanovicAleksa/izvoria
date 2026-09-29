"""
The "Start appliance" state machine: Vast start -> wait for the container -> wait for the agent -> launch the desktop
app in the chosen language. Stop is the reverse. The landing page polls `status()`.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

from deploy.gateway.agent_client import AgentClient, AgentHealth, AgentRefused, AgentUnavailable
from deploy.gateway.config import AGENT_PORT, LANGUAGES, NOVNC_PORT, GatewaySettings
from deploy.gateway.vast_client import InstanceInfo, IVastClient, VastError

logger = logging.getLogger(__name__)

POLL_SECONDS = 3.0
CONTAINER_TIMEOUT_SECONDS = 15 * 60
AGENT_TIMEOUT_SECONDS = 5 * 60
INFO_CACHE_SECONDS = 5.0


class ApplianceState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    GPU_UNAVAILABLE = "gpu_unavailable"
    BOOTING = "booting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True)
class StatusView:
    state: ApplianceState
    language: Optional[str]
    message: str
    error: str = ""

    def as_json(self) -> dict:
        return {"state": self.state.value, "language": self.language, "message": self.message, "error": self.error}


class ApplianceController:
    def __init__(
        self,
        settings: GatewaySettings,
        vast: IVastClient,
        agent: AgentClient,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._vast = vast
        self._agent = agent
        self._clock = clock
        self._sleep = sleep
        self._task: Optional[asyncio.Task] = None
        self._step = ApplianceState.STOPPED
        self._error = ""
        self._scheduling_since: Optional[float] = None
        self._info: Optional[InstanceInfo] = None
        self._info_at = float("-inf")

    # ---------- queries ----------

    async def info(self, fresh: bool = False) -> InstanceInfo:
        if fresh or self._info is None or self._clock() - self._info_at > INFO_CACHE_SECONDS:
            self._info = await self._vast.info()
            self._info_at = self._clock()
        return self._info

    async def agent_address(self) -> Optional[str]:
        return (await self.info()).address(AGENT_PORT)

    async def desktop_address(self) -> Optional[str]:
        info = await self.info()
        return info.address(NOVNC_PORT) if info.running else None

    async def health(self) -> Optional[AgentHealth]:
        info = await self.info()
        address = info.address(AGENT_PORT)
        if not info.running or not address:
            return None
        try:
            return await self._agent.health(address)
        except (AgentUnavailable, AgentRefused):
            return None

    @property
    def transitioning(self) -> bool:
        return self._task is not None and not self._task.done()

    async def status(self) -> StatusView:
        if self.transitioning:
            return self._view(self._step)
        try:
            info = await self.info()
        except VastError as exc:
            return StatusView(ApplianceState.ERROR, None, _MESSAGES[ApplianceState.ERROR], str(exc))
        if not info.running:
            return self._view(ApplianceState.STOPPED)
        health = await self.health()
        if health is None or not health.ready:
            return self._view(ApplianceState.BOOTING)
        return self._view(ApplianceState.BUSY if health.busy else ApplianceState.READY, health.app_language)

    def _view(self, state: ApplianceState, language: Optional[str] = None) -> StatusView:
        return StatusView(state, language, _MESSAGES[state], self._error)

    # ---------- commands ----------

    def start(self, language: str) -> None:
        if language not in LANGUAGES:
            raise ValueError(f"'{language}' is not a language this appliance serves.")
        if self.transitioning:
            return
        self._error = ""
        self._step = ApplianceState.STARTING
        self._task = asyncio.create_task(self._run(self._start(language)))

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._error = ""
        self._step = ApplianceState.STOPPING
        self._task = asyncio.create_task(self._run(self._stop()))

    async def wait(self) -> None:
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self, work: Awaitable[None]) -> None:
        try:
            await work
        except asyncio.CancelledError:
            raise
        except (VastError, AgentUnavailable, AgentRefused, TimeoutError) as exc:
            logger.warning(f"Appliance transition failed: {exc}")
            self._error = str(exc)
        except Exception as exc:
            logger.exception("Appliance transition failed unexpectedly")
            self._error = f"Unexpected error: {exc}"

    async def _start(self, language: str) -> None:
        self._step = ApplianceState.STARTING
        self._scheduling_since = None
        info = await self.info(fresh=True)
        if not info.running:
            await self._vast.start()
            info = await self._wait_for_container()
        self._step = ApplianceState.BOOTING
        address = info.address(AGENT_PORT)
        if not address:
            raise VastError(f"The instance runs but doesn't expose port {AGENT_PORT}; check its port mappings.")
        await self._wait_for_agent(address)
        # An app already running is kept whatever language it started in (it switches languages itself); a refusal,
        # e.g. an unknown language, carries its reason, which becomes the page's error line.
        await self._agent.session(address, language)

    async def _stop(self) -> None:
        self._step = ApplianceState.STOPPING
        info = await self.info(fresh=True)
        address = info.address(AGENT_PORT)
        if info.running and address:
            try:
                await self._agent.shutdown(address)
            except (AgentUnavailable, AgentRefused) as exc:
                logger.warning(f"Agent shutdown skipped: {exc}")
        await self._vast.stop()
        self._info = None

    async def _wait_for_container(self) -> InstanceInfo:
        deadline = self._clock() + CONTAINER_TIMEOUT_SECONDS
        while self._clock() < deadline:
            info = await self.info(fresh=True)
            if info.running:
                return info
            if info.actual_status == "scheduling":
                # Vast: stuck in "scheduling" means the card this instance lives on is rented to someone else.
                self._scheduling_since = self._scheduling_since or self._clock()
                if self._clock() - self._scheduling_since >= self._settings.scheduling_grace_seconds:
                    self._step = ApplianceState.GPU_UNAVAILABLE
            await self._sleep(POLL_SECONDS)
        raise TimeoutError(_MESSAGES[ApplianceState.GPU_UNAVAILABLE] if self._step is ApplianceState.GPU_UNAVAILABLE
                           else "The GPU box didn't start within 15 minutes.")

    async def _wait_for_agent(self, address: str) -> None:
        deadline = self._clock() + AGENT_TIMEOUT_SECONDS
        while self._clock() < deadline:
            try:
                if (await self._agent.health(address)).ready:
                    return
            except (AgentUnavailable, AgentRefused):
                pass
            await self._sleep(POLL_SECONDS)
        raise TimeoutError("The GPU box started but its services didn't come up within 5 minutes.")


_MESSAGES = {
    ApplianceState.STOPPED: "The appliance is off. Choose a language and start it.",
    ApplianceState.STARTING: "Starting the GPU box… this usually takes 1–2 minutes.",
    ApplianceState.GPU_UNAVAILABLE: (
        "The GPU this appliance uses is currently rented by someone else. It will start as soon as it's free; "
        "you can leave this page open or try again later."
    ),
    ApplianceState.BOOTING: "The GPU box is up; starting the appliance's services…",
    ApplianceState.READY: "Ready.",
    ApplianceState.BUSY: "Working (ingesting or writing a report). You can open it to watch.",
    ApplianceState.STOPPING: "Shutting down…",
    ApplianceState.ERROR: "Something went wrong.",
}
