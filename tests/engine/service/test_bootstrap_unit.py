# Canvas: test_bootstrap_unit_v3
# Path: tests/engine/service/test_bootstrap_unit.py

import pytest
from unittest.mock import patch

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
            assert engine._live_deep_research_pipeline is not None
            assert engine._session_memory_api is not None
        finally:
            # The container starts a real BackgroundModelWorker whose telemetry thread polls
            # torch.cuda every 0.5s; left running, it leaks into later tests that patch torch.cuda.
            engine._async_service.shutdown()