"""The GPU boxes: the cheapest that can start is used, and only one ever bills."""

import httpx
import pytest

from deploy.gateway.vast_client import InstanceInfo, IVastClient, VastError
from web_deploy.gateway.config import parse_instance_ids
from web_deploy.gateway.vast_pool import NoGpuAvailable, VastPool, build_pool

QUEUED = "Required resources are currently unavailable, state change queued."


class Box(IVastClient):
    """One Vast instance. `gpu` is "free", "refused" (Vast refuses and queues the start) or "taken" (it schedules)."""

    def __init__(self, gpu="free", status="exited", price=None):
        self.gpu = gpu
        self.price = price
        self.actual = status
        self.intended = "running" if status == "running" else "stopped"
        self.calls = []

    async def start(self):
        self.calls.append("start")
        self.intended = "running"
        if self.gpu == "refused":
            raise VastError(QUEUED)
        self.actual = "running" if self.gpu == "free" else "scheduling"

    async def stop(self):
        self.calls.append("stop")
        self.intended = "stopped"
        self.actual = "exited"

    async def info(self):
        return InstanceInfo(actual_status=self.actual, intended_status=self.intended, price_per_hour=self.price)

    @property
    def bills(self):
        return self.actual == "running" or self.intended == "running"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


def pool(*boxes, grace=60.0):
    clock = Clock()
    return VastPool({1000 + n: box for n, box in enumerate(boxes)}, grace_seconds=grace, clock=clock,
                    sleep=clock.sleep), clock


class TestStarting:
    @pytest.mark.asyncio
    async def test_the_primary_is_used_when_its_gpu_is_free(self):
        primary, spare = Box(), Box()
        p, _ = pool(primary, spare)

        await p.start()

        assert p.active == 1000 and (await p.info()).running
        assert spare.calls == [] and not spare.bills

    @pytest.mark.asyncio
    async def test_a_refused_start_is_cancelled_before_the_spare_starts(self):
        primary, spare = Box(gpu="refused"), Box()
        p, _ = pool(primary, spare)

        await p.start()

        assert p.active == 1001 and (await p.info()).running
        assert primary.calls == ["start", "stop"] and not primary.bills  # Vast's queued start can't fire later

    @pytest.mark.asyncio
    async def test_a_box_left_waiting_for_its_gpu_is_given_up_after_the_grace(self):
        primary, spare = Box(gpu="taken"), Box()
        p, clock = pool(primary, spare, grace=30)

        await p.start()

        assert p.active == 1001
        assert primary.calls == ["start", "stop"] and not primary.bills
        assert 30 <= clock.now < 40

    @pytest.mark.asyncio
    async def test_when_every_gpu_is_taken_nothing_is_left_billing(self, caplog):
        boxes = [Box(gpu="refused"), Box(gpu="taken"), Box(gpu="refused")]
        p, _ = pool(*boxes)

        with pytest.raises(NoGpuAvailable, match="in use by other customers") as refused:
            await p.start()

        assert "1000" not in str(refused.value)  # firms see plain words; the log has the details
        assert "1000: " + QUEUED in caplog.text
        assert not any(box.bills for box in boxes)
        assert p.active == 1000

    @pytest.mark.asyncio
    async def test_a_spare_left_running_is_stopped_before_another_box_starts(self):
        primary, spare = Box(), Box(status="running")
        p, _ = pool(primary, spare)
        p._active = 1000  # e.g. the gateway restarted and wakes the primary

        await p.start()

        assert spare.calls == ["stop"] and primary.bills


class TestCheapestFirst:
    @pytest.mark.asyncio
    async def test_the_cheapest_box_is_woken_even_when_listed_last(self):
        dear, middle, cheap = Box(price=0.40), Box(price=0.35), Box(price=0.32)
        p, _ = pool(dear, middle, cheap)

        await p.start()

        assert p.active == 1002 and cheap.bills
        assert dear.calls == [] and middle.calls == []

    @pytest.mark.asyncio
    async def test_when_the_cheapest_gpu_is_taken_the_next_cheapest_is_woken(self):
        dear, middle, cheap = Box(price=0.40), Box(price=0.35), Box(gpu="refused", price=0.32)
        p, _ = pool(dear, middle, cheap)

        await p.start()

        assert p.active == 1001 and middle.bills
        assert not cheap.bills and dear.calls == []

    @pytest.mark.asyncio
    async def test_a_box_without_a_price_comes_after_every_priced_one(self):
        unpriced, priced = Box(price=None), Box(price=0.90)
        p, _ = pool(unpriced, priced)

        await p.start()

        assert p.active == 1001 and unpriced.calls == []

    @pytest.mark.asyncio
    async def test_equal_prices_keep_the_configured_order(self):
        first, second = Box(price=0.35), Box(price=0.35)
        p, _ = pool(first, second)

        await p.start()

        assert p.active == 1000 and second.calls == []

    @pytest.mark.asyncio
    async def test_a_running_box_is_not_swapped_for_a_cheaper_one(self):
        cheap, running = Box(price=0.30), Box(status="running", price=0.40)
        p, _ = pool(cheap, running)

        assert (await p.info()).running and p.active == 1001
        assert cheap.calls == [] and running.calls == []


class TestOnlyOneBills:
    @pytest.mark.asyncio
    async def test_the_running_box_is_the_one_in_use(self):
        primary, spare = Box(), Box(status="running")
        p, _ = pool(primary, spare)

        assert (await p.info()).running and p.active == 1001
        assert spare.calls == [] and primary.calls == []

    @pytest.mark.asyncio
    async def test_a_second_box_that_wakes_up_is_stopped(self):
        primary, spare = Box(), Box()
        p, _ = pool(primary, spare)
        await p.start()
        spare.intended, spare.actual = "running", "scheduling"  # a start queued long ago finally fires

        assert (await p.info()).running
        assert spare.calls == ["stop"] and not spare.bills and primary.bills

    @pytest.mark.asyncio
    async def test_a_stop_is_not_resent_on_every_status_poll(self):
        primary, spare = Box(), Box()
        p, _ = pool(primary, spare)
        await p.start()
        spare.intended = "running"
        spare.stop = _ignored(spare)  # Vast hasn't caught up yet

        for _ in range(5):
            await p.info()

        assert spare.calls == ["stop"]

    @pytest.mark.asyncio
    async def test_stop_stops_every_box_that_bills(self):
        boxes = [Box(status="running"), Box(), Box(status="running")]
        p, _ = pool(*boxes)

        await p.stop()

        assert [b.calls for b in boxes] == [["stop"], [], ["stop"]]
        assert not any(b.bills for b in boxes)


def _ignored(box):
    async def stop():
        box.calls.append("stop")
    return stop


class TestProduction:
    @pytest.mark.asyncio
    async def test_one_listing_call_describes_every_box(self):
        seen = []

        def vast(request):
            seen.append((request.method, request.url.path))
            return httpx.Response(200, json={"instances": [
                {"id": 1, "actual_status": "exited", "intended_status": "stopped", "dph_total": 0.3211},
                {"id": 2, "actual_status": "running", "intended_status": "running", "public_ipaddr": "203.0.113.9",
                 "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40022"}]}},
                {"id": 99, "actual_status": "running", "intended_status": "running"},  # not ours to manage
            ]})

        p = build_pool("key", (1, 2), 60.0, transport=httpx.MockTransport(vast))
        info = await p.info()

        assert info.running and info.address(22) == "203.0.113.9:40022" and p.active == 2
        assert seen == [("GET", "/api/v1/instances/")]
        assert (await p._infos())[1].price_per_hour == 0.3211 and info.price_per_hour is None

    def test_instance_ids_are_a_comma_list_in_preference_order(self):
        assert parse_instance_ids("55919995, 55823057,55919995") == (55919995, 55823057)
        assert parse_instance_ids("") == ()
        with pytest.raises(RuntimeError):
            parse_instance_ids("559x")


class TestWithTheController:
    @pytest.mark.asyncio
    async def test_a_wake_with_every_gpu_taken_says_why_and_locks_the_boxes_out_again(self, tmp_path):
        from web_deploy.gateway.box import BoxController, BoxState
        from web_deploy.gateway.config import GatewaySettings

        access = tmp_path / "access"
        settings = GatewaySettings(api_secret="k" * 40, database_path=tmp_path / "g.db", static_dir=tmp_path,
                                   box_link="tunnel", box_access_file=access)
        p, clock = pool(Box(gpu="refused"), Box(gpu="taken"))
        box = BoxController(settings, p, httpx.AsyncClient(), clock=clock, sleep=clock.sleep)

        box.start()
        await box.wait()
        status = await box.status()

        assert status.state is BoxState.STOPPED
        assert "in use by other customers" in status.error
        assert access.read_text().strip() == "closed"
