# Canvas: test_background_worker_v2
# Path: tests/core/model_manager/utils/test_background_worker.py

import time
import concurrent.futures
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

    def test_discard_preload_cancels_pending_future(self):
        mock_orchestrator = MagicMock()
        worker = BackgroundModelWorker(mock_orchestrator)

        future = concurrent.futures.Future()
        worker.discard_preload(future, MagicMock())

        assert future.cancelled() is True
        mock_orchestrator.release_model.assert_not_called()
        worker.shutdown()

    def test_discard_preload_releases_already_loaded_model(self):
        mock_orchestrator = MagicMock()
        worker = BackgroundModelWorker(mock_orchestrator)
        loader, loaded = MagicMock(), MagicMock()

        future = concurrent.futures.Future()
        future.set_running_or_notify_cancel()
        future.set_result(loaded)
        worker.discard_preload(future, loader)

        # release_preload, not release_model: the orchestrator has to check nobody has taken the model
        # since, because freeing one that is in use aborts the process instead of raising.
        mock_orchestrator.release_preload.assert_called_once_with(loaded, loader)
        mock_orchestrator.release_model.assert_not_called()
        worker.shutdown()

    def test_discard_preload_releases_running_load_once_it_completes(self):
        mock_orchestrator = MagicMock()
        worker = BackgroundModelWorker(mock_orchestrator)
        loader, loaded = MagicMock(), MagicMock()

        future = concurrent.futures.Future()
        future.set_running_or_notify_cancel()
        worker.discard_preload(future, loader)
        mock_orchestrator.release_preload.assert_not_called()

        future.set_result(loaded)
        mock_orchestrator.release_preload.assert_called_once_with(loaded, loader)
        worker.shutdown()

    def test_discard_preload_ignores_failed_load(self):
        mock_orchestrator = MagicMock()
        worker = BackgroundModelWorker(mock_orchestrator)

        future = concurrent.futures.Future()
        future.set_running_or_notify_cancel()
        future.set_exception(RuntimeError("load failed"))
        worker.discard_preload(future, MagicMock())

        mock_orchestrator.release_preload.assert_not_called()
        worker.shutdown()

    def test_release_all_delegates_to_orchestrator(self):
        mock_orchestrator = MagicMock()
        worker = BackgroundModelWorker(mock_orchestrator)

        worker.release_all()

        mock_orchestrator.release_all.assert_called_once_with({})
        worker.shutdown()
