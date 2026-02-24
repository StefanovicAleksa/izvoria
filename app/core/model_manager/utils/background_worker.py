# filename: background_worker_v1.py
# path: app/core/model_manager/utils/background_worker.py
import concurrent.futures

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader

class BackgroundModelWorker:
    """
    Internal Utility to handle threading for AI model loading.
    Strictly uses max_workers=1 to ensure VRAM allocation is sequential.
    """
    def __init__(self, orchestrator: ModelManagerOrchestrator):
        self._orchestrator = orchestrator
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="OnyxModelWorker"
        )

    def preload_model(self, manifest: ModelManifest, loader: IModelLoader) -> concurrent.futures.Future:
        def _load_task() -> LoadedModel:
            return self._orchestrator.acquire_model(manifest, loader)
        return self._executor.submit(_load_task)

    def unload_model(self, loaded_model: LoadedModel, loader: IModelLoader, flush: bool = True) -> concurrent.futures.Future:
        def _unload_task() -> bool:
            self._orchestrator.release_model(loaded_model, loader, flush=flush)
            return True
        return self._executor.submit(_unload_task)

    def shutdown(self, wait: bool = True):
        self._executor.shutdown(wait=wait)