# Canvas: test_background_worker_v2
# Path: tests/core/model_manager/utils/test_background_worker.py

import time
from unittest.mock import MagicMock
from app.core.model_manager.utils.background_worker import BackgroundModelWorker


class TestBackgroundModelWorker:
    def test_telemetry_daemon_spins_up_and_shuts_down(self):
        mock_orchestrator = MagicMock()

        worker = BackgroundModelWorker(mock_orchestrator)
        assert worker._telemetry_thread.is_alive() is True

        # Let it tick at least once
        time.sleep(0.6)
        mock_orchestrator.record_telemetry.assert_called()

        # Send the shutdown signal
        worker.shutdown(wait=True)

        # CRITICAL FIX: Ensure the test waits for the OS thread scheduler to reap the thread
        worker._telemetry_thread.join(timeout=1.0)

        assert worker._telemetry_thread.is_alive() is False

    def test_preload_model_submits_to_executor(self):
        mock_orchestrator = MagicMock()
        mock_orchestrator.acquire_model.return_value = "LoadedMock"

        worker = BackgroundModelWorker(mock_orchestrator)
        future = worker.preload_model(MagicMock(), MagicMock())

        result = future.result()
        assert result == "LoadedMock"
        worker.shutdown()