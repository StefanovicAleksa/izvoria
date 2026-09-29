from unittest.mock import AsyncMock

import pytest

from deploy.gateway.agent_client import AgentHealth, AgentRefused, AgentUnavailable
from deploy.gateway.config import GatewaySettings
from deploy.gateway.controller import ApplianceController, ApplianceState
from deploy.gateway.vast_client import InstanceInfo, IVastClient

PORTS = {8090: 41001, 6080: 41002}
RUNNING = InstanceInfo("running", "running", "203.0.113.9", PORTS)
STOPPED = InstanceInfo("stopped", "stopped")
SCHEDULING = InstanceInfo("scheduling", "running")
AGENT = "203.0.113.9:41001"


def health(ready=True, busy=False, language=None):
    return AgentHealth(postgres=ready, display=ready, app_language=language, busy=busy)


class ScriptedVast(IVastClient):
    """Returns `infos` in order, repeating the last one."""

    def __init__(self, *infos):
        self.infos = list(infos)
        self.calls = []

    async def start(self):
        self.calls.append("start")

    async def stop(self):
        self.calls.append("stop")

    async def info(self):
        return self.infos.pop(0) if len(self.infos) > 1 else self.infos[0]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def settings():
    return GatewaySettings(password="p" * 12, session_secret="s" * 32, agent_secret="a" * 32,
                           websockify_token="t" * 32, scheduling_grace_seconds=60)


def controller(settings, vast, agent, clock=None):
    clock = clock or Clock()
    return ApplianceController(settings, vast, agent, clock=clock, sleep=clock.sleep)


class TestStart:
    @pytest.mark.asyncio
    async def test_starts_the_box_waits_for_the_agent_then_launches_the_language(self, settings):
        vast = ScriptedVast(STOPPED, STOPPED, InstanceInfo("loading", "running"), RUNNING)
        agent = AsyncMock()
        agent.health.side_effect = [AgentUnavailable("not yet"), health(ready=False), health()]
        c = controller(settings, vast, agent)

        c.start("sr")
        await c.wait()

        assert vast.calls == ["start"]
        assert [call.args for call in agent.health.call_args_list] == [(AGENT,)] * 3
        agent.session.assert_awaited_once_with(AGENT, "sr")

    @pytest.mark.asyncio
    async def test_reports_starting_before_the_task_has_had_a_turn(self, settings):
        c = controller(settings, ScriptedVast(STOPPED), AsyncMock())
        c.start("sr")
        assert (await c.status()).state is ApplianceState.STARTING
        c.stop()
        assert (await c.status()).state is ApplianceState.STOPPING
        await c.wait()

    @pytest.mark.asyncio
    async def test_a_running_box_only_switches_the_session(self, settings):
        agent = AsyncMock()
        agent.health.return_value = health()
        vast = ScriptedVast(RUNNING)
        c = controller(settings, vast, agent)

        c.start("en")
        await c.wait()

        assert vast.calls == []
        agent.session.assert_awaited_once_with(AGENT, "en")

    @pytest.mark.asyncio
    async def test_rejects_languages_the_appliance_does_not_serve(self, settings):
        with pytest.raises(ValueError):
            controller(settings, ScriptedVast(STOPPED), AsyncMock()).start("de")

    @pytest.mark.asyncio
    async def test_stuck_scheduling_is_reported_as_gpu_unavailable(self, settings):
        clock = Clock()
        vast = ScriptedVast(STOPPED, SCHEDULING)
        c = controller(settings, vast, AsyncMock(), clock)
        states = []

        async def watching_sleep(seconds):
            states.append((await c.status()).state)
            await Clock.sleep(clock, seconds)
        c._sleep = watching_sleep

        c.start("sr")
        await c.wait()

        assert states[0] is ApplianceState.STARTING
        assert ApplianceState.GPU_UNAVAILABLE in states
        view = await c.status()
        assert view.state is ApplianceState.STOPPED  # gave up after 15 min, and says why
        assert "rented by someone else" in view.error

    @pytest.mark.asyncio
    async def test_an_agent_refusal_surfaces_as_the_error_line(self, settings):
        agent = AsyncMock()
        agent.health.return_value = health(language="en", busy=True)
        agent.session.side_effect = AgentRefused(400, "'de' is not a language this appliance serves.")
        c = controller(settings, ScriptedVast(RUNNING), agent)

        c.start("sr")
        await c.wait()

        view = await c.status()
        assert (view.state, view.language) == (ApplianceState.BUSY, "en")
        assert view.error == "'de' is not a language this appliance serves."


class TestStop:
    @pytest.mark.asyncio
    async def test_closes_the_app_before_stopping_the_box(self, settings):
        order = []
        vast = ScriptedVast(RUNNING)
        vast.stop = AsyncMock(side_effect=lambda: order.append("vast.stop"))
        agent = AsyncMock()
        agent.shutdown.side_effect = lambda address: order.append(("shutdown", address))
        c = controller(settings, vast, agent)

        c.stop()
        await c.wait()

        assert order == [("shutdown", AGENT), "vast.stop"]

    @pytest.mark.asyncio
    async def test_an_unreachable_agent_does_not_block_the_stop(self, settings):
        vast = ScriptedVast(RUNNING)
        agent = AsyncMock()
        agent.shutdown.side_effect = AgentUnavailable("down")
        c = controller(settings, vast, agent)

        c.stop()
        await c.wait()

        assert vast.calls == ["stop"]


class TestStatus:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("info, agent_health, expected", [
        (STOPPED, None, (ApplianceState.STOPPED, None)),
        (RUNNING, AgentUnavailable("x"), (ApplianceState.BOOTING, None)),
        (RUNNING, health(ready=False), (ApplianceState.BOOTING, None)),
        (RUNNING, health(language="sr"), (ApplianceState.READY, "sr")),
        (RUNNING, health(language="en", busy=True), (ApplianceState.BUSY, "en")),
        (RUNNING, health(language=None), (ApplianceState.READY, None)),
    ])
    async def test_maps_box_and_agent_state(self, settings, info, agent_health, expected):
        agent = AsyncMock()
        if isinstance(agent_health, Exception):
            agent.health.side_effect = agent_health
        else:
            agent.health.return_value = agent_health
        view = await controller(settings, ScriptedVast(info), agent).status()
        assert (view.state, view.language) == expected
