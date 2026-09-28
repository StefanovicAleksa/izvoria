from unittest.mock import AsyncMock, MagicMock

import pytest

from deploy.gateway.agent_client import AgentHealth
from deploy.gateway.idle_monitor import IdleMonitor
from deploy.gateway.vast_client import InstanceInfo

IDLE = 30 * 60


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make(running=True, busy=False, transitioning=False):
    controller = MagicMock()
    controller.transitioning = transitioning
    controller.info = AsyncMock(return_value=InstanceInfo("running" if running else "stopped", "x"))
    controller.health = AsyncMock(return_value=AgentHealth(True, True, "sr", busy))
    clock = Clock()
    return controller, clock, IdleMonitor(controller, IDLE, clock=clock)


class TestIdleMonitor:
    @pytest.mark.asyncio
    async def test_stops_an_idle_unwatched_box_after_the_timeout(self):
        controller, clock, monitor = make()
        clock.now += IDLE - 1
        assert await monitor.check() is False
        clock.now += 1
        assert await monitor.check() is True
        controller.stop.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_never_stops_while_the_appliance_is_working(self):
        controller, clock, monitor = make(busy=True)
        clock.now += 10 * IDLE
        assert await monitor.check() is False
        controller.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_time_counts_from_when_the_work_finished(self):
        controller, clock, monitor = make(busy=True)
        clock.now += 10 * IDLE
        await monitor.check()
        controller.health.return_value = AgentHealth(True, True, "sr", False)
        clock.now += IDLE - 1
        assert await monitor.check() is False
        clock.now += 1
        assert await monitor.check() is True

    @pytest.mark.asyncio
    async def test_never_stops_while_someone_is_connected(self):
        controller, clock, monitor = make()
        with monitor.socket_open():
            clock.now += 10 * IDLE
            assert await monitor.check() is False
        controller.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_never_stops_during_an_upload(self):
        controller, clock, monitor = make()
        with monitor.upload_in_flight():
            clock.now += 10 * IDLE
            assert await monitor.check() is False

    @pytest.mark.asyncio
    async def test_leaves_a_starting_or_stopping_box_alone(self):
        controller, clock, monitor = make(transitioning=True)
        clock.now += 10 * IDLE
        assert await monitor.check() is False
        controller.info.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nothing_to_stop_when_the_box_is_off(self):
        controller, clock, monitor = make(running=False)
        clock.now += 10 * IDLE
        assert await monitor.check() is False
        controller.stop.assert_not_called()
