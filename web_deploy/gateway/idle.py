"""
Stops the GPU box once no firm needs it: nobody has used it for IDLE_MINUTES and it has nothing left to do.

"Nothing to do" is the box's own word (its /health): no question being answered, no report being written, no
ingestion running or waiting. A report or a queued ingestion runs for hours with every browser closed, and stopping
the box mid-run would lose it. Adapted from deploy/gateway/idle_monitor.py; activity now counts across all firms.
"""

import asyncio
import contextlib
import logging
import time
from typing import Callable

from deploy.gateway.vast_client import VastError
from web_deploy.gateway.box import BoxController

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 30.0


class IdleMonitor:
    def __init__(self, box: BoxController, idle_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._box = box
        self._idle_seconds = idle_seconds
        self._clock = clock
        self._last_activity = clock()
        self._open = 0

    def touch(self) -> None:
        self._last_activity = self._clock()

    @contextlib.contextmanager
    def in_flight(self):
        """A proxied request, an event stream or an upload: someone is using the appliance right now."""
        self._open += 1
        self.touch()
        try:
            yield
        finally:
            self._open -= 1
            self.touch()

    async def check(self) -> bool:
        """Returns True when it asked the box to stop."""
        if self._box.transitioning or self._open:
            self.touch()
            return False
        try:
            info = await self._box.info(fresh=True)
        except VastError as exc:
            logger.warning(f"Idle check skipped: {exc}")
            return False
        if not info.running:
            self.touch()
            return False
        health = await self._box.health(fresh=True)
        if health is not None and health.busy:
            # Idle time counts from when the work finishes, not from when the last browser closed
            self.touch()
            return False
        if self._clock() - self._last_activity < self._idle_seconds:
            return False
        logger.info(f"GPU box idle for {self._idle_seconds / 60:.0f} min with nothing to do; stopping it.")
        self._box.stop()
        return True

    async def run_forever(self) -> None:
        while True:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            try:
                await self.check()
            except Exception:
                logger.exception("Idle check failed")
