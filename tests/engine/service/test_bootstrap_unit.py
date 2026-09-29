# Canvas: test_bootstrap_unit_v3
# Path: tests/engine/service/test_bootstrap_unit.py

import pytest
from unittest.mock import Mock, call, patch

from app.engine.exceptions.lifecycle import EngineBusyError
from app.engine.service.bootstrap import OnyxBootstrapper
from app.engine.service.onyx_engine import OnyxEngine


class TestOnyxBootstrapperUnit:

    @patch('app.core.database.service.manager.db_manager')
    @patch('app.features.scanner.service.scanner.FileScanner')
    @patch('app.core.model_manager.data.vram_monitor.PyTorchVramMonitor')
    def test_bootstrapper_builds_engine_successfully(self, mock_vram, mock_scanner, mock_db):
        """
        Validates that the massive dependency injection container can assemble
        all repositories, adapters, APIs, and phases without raising an exception.
        """
        try:
            engine = OnyxBootstrapper.build()
        except Exception as e:
            pytest.fail(f"Bootstrapper failed to assemble dependencies: {e}")

        try:
            assert isinstance(engine, OnyxEngine)
            assert engine._media_pipeline is not None
            assert engine._audio_pipeline is not None
            assert engine._live_pipeline is not None
            assert engine._live_verified_pipeline is not None
            assert engine._live_qa_pipeline is not None
            assert engine._live_reasoning_pipeline is not None
            assert engine._session_memory_api is not None
        finally:
            # The container starts a real BackgroundModelWorker whose telemetry thread polls
            # torch.cuda every 0.5s; left running, it leaks into later tests that patch torch.cuda.
            engine._async_service.shutdown()

class TestSwitchLanguage:
    @pytest.fixture
    def settings(self):
        with patch('app.engine.service.bootstrap.settings') as settings:
            settings.APPLIANCE_LANGUAGE = "en"
            yield settings

    def test_stops_switches_rebuilds_starts_and_remembers_in_that_order(self, settings):
        calls = Mock()
        old = calls.old
        settings.use_language.side_effect = lambda code: calls.use_language(code)
        settings.remember_language.side_effect = lambda code: calls.remember_language(code)
        listener = Mock()
        with patch.object(OnyxBootstrapper, 'build', side_effect=lambda progress_listener: calls.build(progress_listener)):
            rebuilt = OnyxBootstrapper.switch_language(old, "sr", progress_listener=listener)

        assert calls.mock_calls == [
            call.old.require_idle(), call.old.stop(), call.use_language("sr"), call.build(listener),
            call.build().start(), call.remember_language("sr"),
        ]
        assert rebuilt is calls.build.return_value

    def test_a_busy_engine_is_left_alone(self, settings):
        old = Mock()
        old.require_idle.side_effect = EngineBusyError("ingesting")

        with patch.object(OnyxBootstrapper, 'build') as build, pytest.raises(EngineBusyError):
            OnyxBootstrapper.switch_language(old, "sr")

        old.stop.assert_not_called()
        settings.use_language.assert_not_called()
        build.assert_not_called()

    def test_a_failed_rebuild_goes_back_to_the_previous_language_and_remembers_nothing(self, settings):
        with patch.object(OnyxBootstrapper, 'build', side_effect=RuntimeError("no GPU")), \
                pytest.raises(RuntimeError):
            OnyxBootstrapper.switch_language(Mock(), "sr")

        assert settings.use_language.mock_calls == [call("sr"), call("en")]
        settings.remember_language.assert_not_called()

    def test_without_an_engine_it_just_builds_one(self, settings):
        with patch.object(OnyxBootstrapper, 'build') as build:
            rebuilt = OnyxBootstrapper.switch_language(None, "sl")

        settings.use_language.assert_called_once_with("sl")
        build.return_value.start.assert_called_once_with()
        assert rebuilt is build.return_value
