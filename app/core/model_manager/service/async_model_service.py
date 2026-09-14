# Canvas: async_model_service_v3
# Path: app/core/model_manager/service/async_model_service.py
import concurrent.futures

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.utils.error_mapper import map_model_manager_error


class AsyncModelService:
    def __init__(self, worker: BackgroundModelWorker):
        self._worker = worker

    def trigger_preload(self, manifest: ModelManifest, loader: IModelLoader) -> concurrent.futures.Future:
        try:
            return self._worker.preload_model(manifest, loader)
        except Exception as e:
            raise map_model_manager_error(e) from e

    def resolve_preload(self, future: concurrent.futures.Future) -> LoadedModel:
        try:
            return future.result()
        except Exception as e:
            raise map_model_manager_error(e) from e

    def trigger_unload(self, loaded_model: LoadedModel, loader: IModelLoader, flush: bool = True) -> None:
        try:
            # FIX: Synchronous Unload.
            # Bypasses the worker thread pool entirely so we don't deadlock behind a parked preload task.
            self._worker._orchestrator.release_model(loaded_model, loader, flush=flush)
        except Exception as e:
            raise map_model_manager_error(e) from e

    def discard_preload(self, future: concurrent.futures.Future, loader: IModelLoader) -> None:
        """Cancels or releases a preload that will never be resolved by its phase."""
        try:
            self._worker.discard_preload(future, loader)
        except Exception as e:
            raise map_model_manager_error(e) from e

    def release_all(self) -> None:
        """Evicts every resident model. Used by the engine after ingestion or a failed live call."""
        try:
            self._worker.release_all()
        except Exception as e:
            raise map_model_manager_error(e) from e

    def shutdown(self):
        self._worker.shutdown()
