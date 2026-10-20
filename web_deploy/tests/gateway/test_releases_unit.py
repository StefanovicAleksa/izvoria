"""Boxes fetch the current version themselves, and stale boxes are caught up while nobody needs the appliance."""

import json

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from web_deploy.common.signing import sign_service
from web_deploy.gateway import fleet
from web_deploy.gateway.fleet import FleetState
from web_deploy.gateway.releases import CatchUp, ReleaseSettings, UpdateReports, release_routes
from web_deploy.tests.gateway.test_vast_pool_unit import Box, pool

SECRET = "s" * 40
RELEASE = "9abe48d5cd65"


def _gateway(tmp_path, reports=None, maintenance=()):
    releases = tmp_path / "releases"
    releases.mkdir()
    (releases / f"{RELEASE}.tar.gz").write_bytes(b"package bytes")
    (releases / f"{RELEASE}.json").write_text(json.dumps(
        {"release": RELEASE, "sha256": "abc", "size": 13, "provision_digest": "d1"}))
    fleet.save(tmp_path / "fleet.json", FleetState(release=RELEASE, maintenance=frozenset(maintenance)))
    app = Starlette(routes=release_routes(ReleaseSettings(SECRET, releases, tmp_path / "fleet.json", (1000, 1001)),
                                          reports))
    return TestClient(app)


def _signed(method, path):
    return sign_service(SECRET, method, path)


class TestReleaseRoutes:
    def test_a_box_learns_the_current_version(self, tmp_path):
        client = _gateway(tmp_path)

        body = client.get("/box-release/1000", headers=_signed("GET", "/box-release/1000")).json()

        assert body == {"release": RELEASE, "sha256": "abc", "size": 13, "provision_digest": "d1",
                        "maintenance": False}

    def test_asking_is_noted_for_the_catch_up(self, tmp_path):
        reports = UpdateReports()
        client = _gateway(tmp_path, reports)
        client.get("/box-release/1001", headers=_signed("GET", "/box-release/1001"))
        assert reports.asked == {1001}

    def test_a_box_started_only_to_be_updated_is_told_so(self, tmp_path):
        client = _gateway(tmp_path, maintenance=(1001,))
        assert client.get("/box-release/1001", headers=_signed("GET", "/box-release/1001")).json()["maintenance"]

    def test_the_package_is_served(self, tmp_path):
        client = _gateway(tmp_path)
        path = "/box-release/1000/package"
        assert client.get(path, headers=_signed("GET", path)).content == b"package bytes"

    def test_an_applied_version_is_recorded(self, tmp_path):
        client = _gateway(tmp_path)
        path = f"/box-release/1001/applied/{RELEASE}"

        assert client.post(path, headers=_signed("POST", path)).status_code == 200

        assert fleet.load(tmp_path / "fleet.json").boxes == {1001: RELEASE}

    def test_a_failure_is_noted_for_the_catch_up(self, tmp_path):
        reports = UpdateReports()
        client = _gateway(tmp_path, reports)
        path = "/box-release/1000/failed"

        client.post(path, headers=_signed("POST", path))

        assert reports.failed == {1000}

    @pytest.mark.parametrize("method,path", [("GET", "/box-release/1000"), ("GET", "/box-release/1000/package"),
                                             ("POST", f"/box-release/1000/applied/{RELEASE}")])
    def test_nothing_without_the_signature(self, tmp_path, method, path):
        client = _gateway(tmp_path)
        assert client.request(method, path).status_code == 401
        # A signature for another path doesn't open this one
        assert client.request(method, path, headers=_signed(method, "/box-release/1001")).status_code == 401

    def test_an_unknown_box_gets_nothing(self, tmp_path):
        client = _gateway(tmp_path)
        assert client.get("/box-release/999", headers=_signed("GET", "/box-release/999")).status_code == 404

    def test_no_package_no_answer(self, tmp_path):
        client = _gateway(tmp_path)
        (tmp_path / "releases" / f"{RELEASE}.tar.gz").unlink()
        assert client.get("/box-release/1000", headers=_signed("GET", "/box-release/1000")).status_code == 404

    def test_an_applied_name_that_is_not_a_release_is_refused(self, tmp_path):
        client = _gateway(tmp_path)
        path = "/box-release/1000/applied/..%2Fetc"
        assert client.post(path, headers=_signed("POST", path)).status_code in (400, 404)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _catch_up(tmp_path, boxes, state, on_poll=None, timeout=600.0, contact=300.0):
    p, _ = pool(*boxes)
    fleet.save(tmp_path / "fleet.json", state)
    clock = Clock()
    reports = UpdateReports()

    async def sleep(seconds):
        clock.now += seconds
        if on_poll:
            on_poll(reports)

    catch_up = CatchUp(p, tmp_path / "fleet.json", [1000 + n for n in range(len(boxes))], 1200,
                       reports=reports, timeout_seconds=timeout, contact_seconds=contact, clock=clock, sleep=sleep)
    return catch_up, p, reports


class TestCatchUp:
    @pytest.mark.asyncio
    async def test_a_stale_box_is_started_updated_and_stopped(self, tmp_path):
        stale = Box()
        seen_marked = []

        def box_reports(_reports):
            seen_marked.append(1000 in fleet.load(tmp_path / "fleet.json").maintenance)
            state = fleet.load(tmp_path / "fleet.json")  # the box's self_update posts /applied
            fleet.save(tmp_path / "fleet.json", FleetState(state.release, {**state.boxes, 1000: "new"},
                                                           state.maintenance))
        catch_up, _, _ = _catch_up(tmp_path, [stale, Box()],
                                   FleetState(release="new", boxes={1001: "new"}), on_poll=box_reports)

        assert await catch_up.round() == "updated"

        assert stale.calls == ["start", "stop"] and not stale.bills
        assert seen_marked == [True]  # the pool left it alone meanwhile
        assert fleet.load(tmp_path / "fleet.json").maintenance == frozenset()

    @pytest.mark.asyncio
    async def test_nothing_is_started_while_the_appliance_is_in_use(self, tmp_path):
        stale, in_use = Box(), Box(status="running")
        catch_up, _, _ = _catch_up(tmp_path, [stale, in_use], FleetState(release="new", boxes={1001: "new"}))

        assert await catch_up.round() is None
        assert stale.calls == []

    @pytest.mark.asyncio
    async def test_a_box_whose_gpu_is_taken_is_left_for_the_next_round(self, tmp_path):
        taken, other_stale = Box(gpu="refused"), Box(gpu="refused")
        catch_up, _, _ = _catch_up(tmp_path, [taken, other_stale], FleetState(release="new"))

        assert await catch_up.round() == "every stale box's GPU is taken"
        assert not taken.bills and not other_stale.bills
        assert fleet.load(tmp_path / "fleet.json").maintenance == frozenset()

    @pytest.mark.asyncio
    async def test_a_box_that_reports_a_failure_is_stopped_at_once(self, tmp_path):
        stale = Box()
        catch_up, _, _ = _catch_up(tmp_path, [stale], FleetState(release="new"),
                                   on_poll=lambda reports: reports.failed.add(1000))

        assert (await catch_up.round()).startswith("failed")
        assert stale.calls == ["start", "stop"]

    @pytest.mark.asyncio
    async def test_a_box_that_never_reports_is_stopped_after_the_timeout(self, tmp_path):
        stale = Box()
        catch_up, _, _ = _catch_up(tmp_path, [stale], FleetState(release="new"), timeout=60,
                                   on_poll=lambda reports: reports.asked.add(1000))

        assert await catch_up.round() == "timed out"
        assert stale.calls == ["start", "stop"] and not stale.bills

    @pytest.mark.asyncio
    async def test_a_box_that_never_asks_is_stopped_soon_and_left_until_the_next_deploy(self, tmp_path):
        stale = Box()
        catch_up, _, _ = _catch_up(tmp_path, [stale], FleetState(release="new"), timeout=3600, contact=300)

        assert (await catch_up.round()).startswith("never asked")
        assert stale.calls == ["start", "stop"]
        assert await catch_up.round() is None and stale.calls == ["start", "stop"]  # not retried for this version

        fleet.save(tmp_path / "fleet.json", FleetState(release="newer"))
        assert catch_up.stale() == [1000]  # the next deploy tries it again

    @pytest.mark.asyncio
    async def test_without_a_known_version_nothing_is_stale(self, tmp_path):
        stale = Box()
        catch_up, _, _ = _catch_up(tmp_path, [stale], FleetState())

        assert await catch_up.round() is None and stale.calls == []
