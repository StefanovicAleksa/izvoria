# Canvas: background_worker_v3
# Path: app/core/model_manager/utils/background_worker.py
import concurrent.futures
import threading

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader


class BackgroundModelWorker:
    """
    Internal Utility to handle threading for AI model loading.
    Includes an active Telemetry Daemon to track VRAM spikes.
    """

    def __init__(self, orchestrator: ModelManagerOrchestrator):
        self._orchestrator = orchestrator
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="OnyxModelWorker"
        )

        # Start the Telemetry Daemon
        self._telemetry_stop_event = threading.Event()
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_loop,
            name="OnyxVramTelemetry",
            daemon=True
        )
        self._telemetry_thread.start()

    def _telemetry_loop(self):
        """Polls the Orchestrator every 0.5s to record activation spikes."""
        # CRITICAL FIX: Use event.wait(timeout) instead of time.sleep().
        # This allows the thread to wake up instantly and exit when shutdown() is called.
        while not self._telemetry_stop_event.wait(0.5):
            self._orchestrator.record_telemetry()

    def preload_model(self, manifest: ModelManifest, loader: IModelLoader) -> concurrent.futures.Future:
        def _load_task() -> LoadedModel:
            return self._orchestrator.acquire_model(manifest, loader)

        return self._executor.submit(_load_task)

    def unload_model(self, loaded_model: LoadedModel, loader: IModelLoader,
                     flush: bool = True) -> concurrent.futures.Future:
        def _unload_task() -> bool:
            self._orchestrator.release_model(loaded_model, loader, flush=flush)
            return True

        return self._executor.submit(_unload_task)

    def shutdown(self, wait: bool = True):
        # Stop the daemon
        self._telemetry_stop_event.set()

        # Ensure the thread is actually reaped before proceeding
        if self._telemetry_thread.is_alive():
            self._telemetry_thread.join(timeout=1.0)

        self._executor.shutdown(wait=wait)