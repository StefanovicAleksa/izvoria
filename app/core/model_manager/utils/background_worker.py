# Canvas: background_worker_v5
# Path: app/core/model_manager/utils/background_worker.py
import concurrent.futures
import logging
import threading

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader

logger = logging.getLogger(__name__)


class BackgroundModelWorker:
    def __init__(self, orchestrator: ModelManagerOrchestrator):
        self._orchestrator = orchestrator
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="OnyxModelWorker"
        )

        self._telemetry_stop_event = threading.Event()
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_loop,
            name="OnyxVramTelemetry",
            daemon=True
        )
        self._telemetry_thread.start()

    def _telemetry_loop(self):
        while not self._telemetry_stop_event.wait(0.5):
            self._orchestrator.record_telemetry()

    def preload_model(self, manifest: ModelManifest, loader: IModelLoader) -> concurrent.futures.Future:
        def _load_task() -> LoadedModel:
            # FIX: Explicitly instruct the Orchestrator to wait, avoiding fatal evictions
            return self._orchestrator.acquire_model(manifest, loader, wait_for_clear=True)

        return self._executor.submit(_load_task)

    def unload_model(self, loaded_model: LoadedModel, loader: IModelLoader,
                     flush: bool = True) -> concurrent.futures.Future:
        def _unload_task() -> bool:
            self._orchestrator.release_model(loaded_model, loader, flush=flush)
            return True

        return self._executor.submit(_unload_task)

    def discard_preload(self, future: concurrent.futures.Future, loader: IModelLoader) -> None:
        """
        Abandons a preload nobody is going to resolve (e.g. the pipeline halted early).
        A queued task is cancelled outright; a running or finished one has its model
        released as soon as the load completes, so it can never stay resident.
        """
        if future.cancel():
            return

        def _release_when_loaded(done: concurrent.futures.Future) -> None:
            if done.cancelled() or done.exception() is not None:
                return
            try:
                # release_PRELOAD, not release_model: by the time this callback runs the model may
                # have been handed to somebody else, and freeing it under them aborts the process
                # inside llama.cpp rather than raising. The orchestrator makes that check atomically.
                self._orchestrator.release_preload(done.result(), loader)
            except Exception as e:
                logger.error(f"Failed to release discarded preload: {e}")

        future.add_done_callback(_release_when_loaded)

    def release_all(self) -> None:
        self._orchestrator.release_all({})

    def shutdown(self, wait: bool = True):
        self._telemetry_stop_event.set()
        if self._telemetry_thread.is_alive():
            self._telemetry_thread.join(timeout=1.0)
        self._executor.shutdown(wait=wait)
