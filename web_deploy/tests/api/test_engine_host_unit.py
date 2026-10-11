"""
EngineHost without the appliance: the `app.*` pieces it imports in start() are injected as fakes, so what is tested
here is its own logic (one engine per firm, built once, a failed build leaving nothing behind, the GPU handoff).
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from web_deploy.api.config import ApiSettings
from web_deploy.api.engine_host import EngineHost
from web_deploy.api.engine_port import ApplianceIngesting


@pytest.fixture
def settings(tmp_path: Path) -> ApiSettings:
    return ApiSettings(secret="s" * 32, tenants_root=tmp_path / "tenants", media_cache_root=tmp_path / "cache",
                       control_database_url="postgresql://unused", firm_pool_size=2, firm_pool_overflow=1)


@pytest.fixture
def app_settings():
    return MagicMock(POSTGRES_USER="onyx", POSTGRES_PASSWORD="pw", POSTGRES_SERVER="db", POSTGRES_PORT=5432)


@pytest.fixture
def host(settings, app_settings):
    host = EngineHost(settings, progress_listener=MagicMock(name="listener"))
    host._app = dict(settings=app_settings, DatabaseManager=MagicMock(name="DatabaseManager"),
                     IzvoriaBootstrapper=MagicMock(name="IzvoriaBootstrapper"))
    host._kit = MagicMock(name="kit")
    host._ready = True
    with patch.object(EngineHost, "_ensure_database") as ensure:
        host.ensure_database = ensure
        yield host


class TestFirmEngines:
    def test_a_firm_engine_gets_its_own_database_vault_and_reports_over_the_shared_kit(self, host, settings):
        engine = host.engine("legal-demo")

        root = settings.tenants_root / "legal-demo"
        host.ensure_database.assert_called_once_with("onyx_firm_legal_demo")
        host._app["DatabaseManager"].assert_called_once_with(
            url="postgresql+psycopg2://onyx:pw@db:5432/onyx_firm_legal_demo", pool_size=2, max_overflow=1)
        host._app["IzvoriaBootstrapper"].build.assert_called_once_with(
            progress_listener=host._progress_listener, database=host._app["DatabaseManager"].return_value,
            vault_dir=root / "vault", reports_dir=root / "reports", kit=host._kit)
        assert engine is host._app["IzvoriaBootstrapper"].build.return_value
        engine.start.assert_called_once_with()
        assert all((root / sub).is_dir() for sub in ("vault", "reports", "uploads"))

    def test_a_firm_engine_is_built_once_and_reused(self, host):
        assert host.engine("legal-demo") is host.engine("legal-demo")

        host._app["IzvoriaBootstrapper"].build.assert_called_once()

    def test_two_firms_get_two_engines(self, host):
        host._app["IzvoriaBootstrapper"].build.side_effect = [MagicMock(name="a"), MagicMock(name="b")]

        assert host.engine("firm-a") is not host.engine("firm-b")

    def test_concurrent_first_requests_for_one_firm_build_one_engine(self, host):
        def slow_build(**_):
            time.sleep(0.05)
            return MagicMock(name="engine")

        host._app["IzvoriaBootstrapper"].build.side_effect = slow_build
        engines = []
        threads = [threading.Thread(target=lambda: engines.append(host.engine("legal-demo"))) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert host._app["IzvoriaBootstrapper"].build.call_count == 1
        assert len(engines) == 4 and all(e is engines[0] for e in engines)

    def test_a_firm_whose_engine_fails_to_start_frees_its_pool_and_is_retried_next_time(self, host):
        database = host._app["DatabaseManager"].return_value
        broken = MagicMock(name="broken")
        broken.start.side_effect = RuntimeError("migration failed")
        healthy = MagicMock(name="healthy")
        host._app["IzvoriaBootstrapper"].build.side_effect = [broken, healthy]

        with pytest.raises(RuntimeError, match="migration failed"):
            host.engine("legal-demo")
        database.dispose.assert_called_once_with()

        assert host.engine("legal-demo") is healthy

    def test_a_failed_build_frees_the_pool_too(self, host):
        host._app["IzvoriaBootstrapper"].build.side_effect = ValueError("bad wiring")

        with pytest.raises(ValueError):
            host.engine("legal-demo")

        host._app["DatabaseManager"].return_value.dispose.assert_called_once_with()

    def test_an_invalid_firm_never_reaches_the_database(self, host):
        with pytest.raises(ValueError):
            host.engine("../etc")

        host.ensure_database.assert_not_called()

    def test_no_engine_before_the_shared_models_are_resident(self, host):
        host._ready = False

        with pytest.raises(RuntimeError, match="still starting"):
            host.engine("legal-demo")


class TestBusyAndStop:
    def test_busy_adds_up_every_firm(self, host):
        a, b = MagicMock(name="a"), MagicMock(name="b")
        a.busy_snapshot.return_value = {"inflight": 2, "reports_running": 0}
        b.busy_snapshot.return_value = {"inflight": 1, "reports_running": 1}
        host._app["IzvoriaBootstrapper"].build.side_effect = [a, b]
        host.engine("firm-a")
        host.engine("firm-b")

        assert host.busy() == {"inflight": 3, "reports_running": 1}

    def test_stop_stops_every_firm_then_frees_the_shared_models(self, host):
        engine = host.engine("legal-demo")
        engine.stop.side_effect = RuntimeError("pool already gone")
        kit = host._kit

        host.stop()

        engine.stop.assert_called_once_with()
        kit.shutdown.assert_called_once_with()
        assert host.ready is False and host._kit is None


class TestGpuHandoff:
    def test_pause_refuses_new_work_and_frees_the_live_models(self, host):
        host.pause(poll_seconds=0.01)

        assert host.paused is True
        host._kit.orchestrator.shutdown_all.assert_called_once_with({})
        with pytest.raises(ApplianceIngesting):
            with host.live():
                pass

    def test_pause_waits_for_work_already_admitted(self, host):
        admitted, release = threading.Event(), threading.Event()

        def work():
            with host.live():
                admitted.set()
                release.wait(5)

        worker = threading.Thread(target=work)
        worker.start()
        admitted.wait(5)
        pauser = threading.Thread(target=host.pause, kwargs={"poll_seconds": 0.01})
        pauser.start()
        time.sleep(0.05)

        host._kit.orchestrator.shutdown_all.assert_not_called()
        release.set()
        pauser.join(5)
        worker.join(5)
        host._kit.orchestrator.shutdown_all.assert_called_once_with({})

    def test_resume_loads_the_models_before_admitting_work(self, host):
        host.pause(poll_seconds=0.01)
        order = []
        host._kit.warm_up.side_effect = lambda: order.append(("warm_up", host.paused)) or []

        host.resume()

        assert order == [("warm_up", True)]
        assert host.paused is False
        with host.live():
            pass

    def test_pause_and_resume_are_idempotent(self, host):
        host.resume()
        host._kit.warm_up.assert_not_called()

        host.pause(poll_seconds=0.01)
        host.pause(poll_seconds=0.01)
        host._kit.orchestrator.shutdown_all.assert_called_once()
