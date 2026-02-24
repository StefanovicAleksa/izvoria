# filename: async_model_service_v1.py
# path: app/core/model_manager/service/async_model_service.py
import concurrent.futures

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.utils.error_mapper import map_model_manager_error


class AsyncModelService:
    """
    Service layer wrapper for background model operations.
    Consumed by the Orchestrator to safely trigger pre-loads and manage errors.
    """
    def __init__(self, worker: BackgroundModelWorker):
        self._worker = worker

    def trigger_preload(self, manifest: ModelManifest, loader: IModelLoader) -> concurrent.futures.Future:
        """Submits a model for background loading."""
        try:
            return self._worker.preload_model(manifest, loader)
        except Exception as e:
            raise map_model_manager_error(e) from e

    def resolve_preload(self, future: concurrent.futures.Future) -> LoadedModel:
        """
        Blocks until the background load is complete.
        Critically, intercepts raw thread exceptions (e.g., CUDA OOM)
        and maps them to strict Domain Exceptions.
        """
        try:
            # .result() will re-raise any exception that occurred inside the background thread
            return future.result()
        except Exception as e:
            # Translate the raw PyTorch/Thread error into OutOfVramError, ModelLoadError, etc.
            raise map_model_manager_error(e) from e

    def trigger_unload(self, loaded_model: LoadedModel, loader: IModelLoader, flush: bool = True) -> None:
        """Submits a model for background unloading."""
        try:
            # We don't typically need to wait for an unload to finish, just fire and forget
            self._worker.unload_model(loaded_model, loader, flush=flush)
        except Exception as e:
            raise map_model_manager_error(e) from e

    def shutdown(self):
        """Safely spins down the background resources."""
        self._worker.shutdown()