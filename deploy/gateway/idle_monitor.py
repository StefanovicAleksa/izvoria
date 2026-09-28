import asyncio
import contextlib
import logging
import time
from typing import Callable

from deploy.gateway.controller import ApplianceController
from deploy.gateway.vast_client import VastError

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 30.0


class IdleMonitor:
    """
    Stops the GPU box once nobody is looking at it AND it has nothing to do. The GPU/CPU check is the one that
    matters: an ingestion or a report runs for hours with the browser closed, and stopping it mid-run loses the run.
    """

    def __init__(self, controller: ApplianceController, idle_seconds: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._controller = controller
        self._idle_seconds = idle_seconds
        self._clock = clock
        self._last_activity = clock()
        self._sockets = 0
        self._uploads = 0

    def touch(self) -> None:
        self._last_activity = self._clock()

    @contextlib.contextmanager
    def socket_open(self):
        self._sockets += 1
        self.touch()
        try:
            yield
        finally:
            self._sockets -= 1
            self.touch()

    @contextlib.contextmanager
    def upload_in_flight(self):
        self._uploads += 1
        self.touch()
        try:
            yield
        finally:
            self._uploads -= 1
            self.touch()

    async def check(self) -> bool:
        """Returns True when it asked the controller to stop the box."""
        if self._controller.transitioning:
            self.touch()
            return False
        if self._sockets or self._uploads:
            self.touch()
            return False
        try:
            info = await self._controller.info(fresh=True)
        except VastError as exc:
            logger.warning(f"Idle check skipped: {exc}")
            return False
        if not info.running:
            self.touch()
            return False
        health = await self._controller.health()
        if health is not None and health.busy:
            # Idle time counts from when the work finishes, not from when the tester closed the tab.
            self.touch()
            return False
        if self._clock() - self._last_activity < self._idle_seconds:
            return False
        logger.info(f"GPU box idle for {self._idle_seconds / 60:.0f} min with nobody connected; stopping it.")
        self._controller.stop()
        return True

    async def run_forever(self) -> None:
        while True:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            try:
                await self.check()
            except Exception:
                logger.exception("Idle check failed")
