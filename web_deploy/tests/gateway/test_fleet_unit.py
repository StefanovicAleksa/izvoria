"""The fleet file: which version each GPU box holds, and which boxes an update is working on."""


import httpx
import pytest

from deploy.gateway.vast_client import InstanceInfo
from web_deploy.gateway import fleet
from web_deploy.gateway.config import GatewaySettings
from web_deploy.gateway.fleet import FleetState
from web_deploy.gateway.fleet_cli import parser, run
from web_deploy.tests.gateway.test_vast_pool_unit import Box, pool


class TestFleetFile:
    def test_a_saved_state_reads_back(self, tmp_path):
        path = tmp_path / "fleet.json"
        state = FleetState(release="abc123", boxes={1: "abc123", 2: "old"}, maintenance=frozenset({2}))

        fleet.save(path, state)

        assert fleet.load(path) == state
        assert path.stat().st_mode & 0o777 == 0o644  # the gateway reads it as another user

    @pytest.mark.parametrize("content", [None, "not json", '{"boxes": {"x": "1"}}', "[]"])
    def test_a_missing_or_broken_file_means_no_versions(self, tmp_path, content):
        path = tmp_path / "fleet.json"
        if content is not None:
            path.write_text(content)

        assert fleet.load(path) == FleetState()

    def test_every_box_is_current_while_no_version_is_known(self):
        assert FleetState(boxes={1: "old"}).current(1) and FleetState().current(7)

    def test_a_box_is_current_only_at_the_current_version(self):
        state = FleetState(release="new", boxes={1: "new", 2: "old"})
        assert state.current(1) and not state.current(2) and not state.current(3)


class TestPoolWithVersions:
    @pytest.mark.asyncio
    async def test_an_up_to_date_box_is_woken_before_a_cheaper_stale_one(self):
        stale, current = Box(price=0.30), Box(price=0.40)
        p, _ = pool(stale, current)
        p._fleet = lambda: FleetState(release="new", boxes={1000: "old", 1001: "new"})

        await p.start()

        assert p.active == 1001 and stale.calls == []

    @pytest.mark.asyncio
    async def test_a_stale_box_is_used_when_no_up_to_date_one_can_start(self):
        stale, current = Box(price=0.30), Box(gpu="refused", price=0.40)
        p, _ = pool(stale, current)
        p._fleet = lambda: FleetState(release="new", boxes={1000: "old", 1001: "new"})

        await p.start()

        assert p.active == 1000 and stale.bills and not current.bills

    @pytest.mark.asyncio
    async def test_a_box_being_updated_is_never_started(self):
        updating, other = Box(price=0.30), Box(price=0.40)
        p, _ = pool(updating, other)
        p._fleet = lambda: FleetState(maintenance=frozenset({1000}))

        await p.start()

        assert p.active == 1001 and updating.calls == []

    @pytest.mark.asyncio
    async def test_a_box_being_updated_is_neither_kept_nor_stopped_nor_reported(self):
        in_use, updating = Box(status="running"), Box(status="running")
        p, clock = pool(in_use, updating)
        p._fleet = lambda: FleetState(maintenance=frozenset({1001}))

        assert (await p.info()).running and p.active == 1000
        await p.stop()

        assert updating.calls == [] and updating.bills  # the update stops it itself
        assert in_use.calls == ["stop"]

    @pytest.mark.asyncio
    async def test_with_every_box_being_updated_none_is_in_use(self):
        updating = Box(status="running")
        p, _ = pool(updating)
        p._fleet = lambda: FleetState(maintenance=frozenset({1000}))

        info = await p.info()

        assert not info.running and updating.calls == []


def _settings(tmp_path, ids=(1000, 1001)):
    return GatewaySettings(api_secret="k" * 40, database_path=tmp_path / "g.db", static_dir=tmp_path,
                           vast_instance_ids=tuple(ids), fleet_file=tmp_path / "fleet.json")


async def _cli(tmp_path, p, *argv, http=None):
    return await run(parser().parse_args(list(argv)), _settings(tmp_path), p, http)


class TestFleetCli:
    @pytest.mark.asyncio
    async def test_status_describes_every_box_with_its_version_and_ssh_address(self, tmp_path):
        first, second = Box(price=0.32), Box(status="running", price=0.40)
        p, _ = pool(first, second)
        fleet.save(tmp_path / "fleet.json", FleetState(release="new", boxes={1001: "new"}))

        async def described():
            return {1000: InstanceInfo("exited", "stopped", "203.0.113.9", {22: 40022}, price_per_hour=0.32),
                    1001: InstanceInfo("running", "running", "203.0.113.9", {22: 40023}, price_per_hour=0.40)}
        p.describe = described

        result = await _cli(tmp_path, p, "status")

        assert result["release"] == "new"
        assert result["boxes"][0] == {"id": 1000, "actual_status": "exited", "intended_status": "stopped",
                                      "price_per_hour": 0.32, "ssh_host": "203.0.113.9", "ssh_port": 40022,
                                      "release": "", "current": False, "maintenance": False}
        assert result["boxes"][1]["current"] is True

    @pytest.mark.asyncio
    async def test_a_box_must_be_under_maintenance_before_it_is_started(self, tmp_path):
        box = Box()
        p, _ = pool(box, Box())

        with pytest.raises(SystemExit):
            await _cli(tmp_path, p, "start", "1000")
        assert box.calls == []

        await _cli(tmp_path, p, "maintenance", "add", "1000")
        result = await _cli(tmp_path, p, "start", "1000")

        assert result == {"id": 1000, "started": True, "reason": ""} and box.bills

    @pytest.mark.asyncio
    async def test_a_refused_start_is_reported_and_left_stopped(self, tmp_path):
        box = Box(gpu="refused")
        p, _ = pool(box, Box())
        await _cli(tmp_path, p, "maintenance", "add", "1000")

        result = await _cli(tmp_path, p, "start", "1000")

        assert result["started"] is False and "unavailable" in result["reason"] and not box.bills

    @pytest.mark.asyncio
    async def test_releases_and_maintenance_are_recorded(self, tmp_path):
        p, _ = pool(Box(), Box())

        await _cli(tmp_path, p, "set-release", "new")
        await _cli(tmp_path, p, "record", "1000", "new")
        await _cli(tmp_path, p, "maintenance", "add", "1001")
        await _cli(tmp_path, p, "maintenance", "remove", "1001")

        assert fleet.load(tmp_path / "fleet.json") == FleetState(release="new", boxes={1000: "new"})

    @pytest.mark.asyncio
    async def test_an_unknown_box_is_refused(self, tmp_path):
        p, _ = pool(Box(), Box())
        with pytest.raises(SystemExit):
            await _cli(tmp_path, p, "record", "999", "new")

    @pytest.mark.asyncio
    async def test_health_reads_the_box_in_use_through_its_tunnel(self, tmp_path):
        p, _ = pool(Box())

        def handler(request):
            assert str(request.url) == "http://127.0.0.1:18095/health"
            return httpx.Response(200, json={"ready": True, "busy": False, "release": "new"})
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        result = await _cli(tmp_path, p, "health", http=http)

        assert result == {"reachable": True, "ready": True, "busy": False, "release": "new"}

    @pytest.mark.asyncio
    async def test_an_unreachable_box_is_reported_not_raised(self, tmp_path):
        p, _ = pool(Box())

        def handler(request):
            raise httpx.ConnectError("refused")
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        assert (await _cli(tmp_path, p, "health", http=http))["reachable"] is False
