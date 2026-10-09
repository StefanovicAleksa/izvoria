import asyncio

import httpx
import pytest

from deploy.gateway.vast_client import FakeVastClient
from web_deploy.gateway.box import BoxController, BoxState
from web_deploy.gateway.config import GatewaySettings
from web_deploy.gateway.idle import IdleMonitor


def settings(tmp_path):
    return GatewaySettings(api_secret="k" * 40, database_path=tmp_path / "g.db",
                           static_dir=tmp_path, fake_box_host="127.0.0.1", idle_minutes=1)


def health_client(body):
    def handler(request):
        assert request.url.path == "/health"
        return httpx.Response(200, json=body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _no_sleep(_):
    await asyncio.sleep(0)


class TestBoxController:
    @pytest.mark.asyncio
    async def test_start_wakes_the_box_and_waits_for_the_models(self, tmp_path):
        vast = FakeVastClient("127.0.0.1", (8095,))
        replies = iter([{"ready": False}, {"ready": False}, {"ready": True, "busy": False}])
        http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=next(replies))))
        box = BoxController(settings(tmp_path), vast, http, sleep=_no_sleep)

        assert (await box.status()).state is BoxState.STOPPED
        box.start()
        await box.wait()

        assert (await box.status()).state is BoxState.READY
        assert await box.api_address() == "127.0.0.1:8095"

    @pytest.mark.asyncio
    async def test_a_model_load_failure_is_shown(self, tmp_path):
        vast = FakeVastClient("127.0.0.1", (8095,))
        await vast.start()
        box = BoxController(settings(tmp_path), vast,
                            health_client({"ready": False, "starting_error": "llama-server unreachable"}))
        status = await box.status()
        assert status.state is BoxState.ERROR
        assert "llama-server" in status.error

    @pytest.mark.asyncio
    async def test_stop(self, tmp_path):
        vast = FakeVastClient("127.0.0.1", (8095,))
        await vast.start()
        box = BoxController(settings(tmp_path), vast, health_client({"ready": True}))
        box.stop()
        await box.wait()
        assert vast.stops == 1


class TestIdleMonitor:
    async def _box(self, tmp_path, health):
        vast = FakeVastClient("127.0.0.1", (8095,))
        await vast.start()
        return vast, BoxController(settings(tmp_path), vast, health_client(health))

    @pytest.mark.asyncio
    async def test_stops_an_idle_box_after_the_timeout(self, tmp_path):
        vast, box = await self._box(tmp_path, {"ready": True, "busy": False})
        now = [0.0]
        idle = IdleMonitor(box, 60, clock=lambda: now[0])
        assert await idle.check() is False
        now[0] = 61
        assert await idle.check() is True
        await box.wait()
        assert vast.stops == 1

    @pytest.mark.asyncio
    async def test_never_stops_while_the_box_has_work(self, tmp_path):
        vast, box = await self._box(tmp_path, {"ready": True, "busy": True, "queued": 2})
        now = [0.0]
        idle = IdleMonitor(box, 60, clock=lambda: now[0])
        now[0] = 1000
        assert await idle.check() is False

    @pytest.mark.asyncio
    async def test_never_stops_while_a_request_is_open(self, tmp_path):
        vast, box = await self._box(tmp_path, {"ready": True, "busy": False})
        now = [0.0]
        idle = IdleMonitor(box, 60, clock=lambda: now[0])
        with idle.in_flight():
            now[0] = 1000
            assert await idle.check() is False
