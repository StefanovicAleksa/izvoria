# filename: orchestrator_v3.py
# path: app/core/model_manager/service/orchestrator.py
import uuid
import threading
from typing import Dict

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IVramMonitor, IModelLoader
from app.core.model_manager import exceptions as exc


class ModelManagerOrchestrator:
    """
    The strict VRAM bouncer and lifecycle manager for AI Models.
    Ensures that GPU memory limits are never exceeded during pipeline execution.
    Now strictly Thread-Safe to support background model preloading.
    """

    # CUDA requires a base overhead just to initialize the context on the GPU.
    VRAM_SAFETY_BUFFER_MB = 500.0

    def __init__(self, vram_monitor: IVramMonitor):
        self._monitor = vram_monitor
        self._active_models: Dict[uuid.UUID, LoadedModel] = {}
        # FIXED: Use RLock (Re-entrant Lock) to prevent self-deadlocks
        # when class methods (like release_all) call other locked methods (like release_model).
        self._lock = threading.RLock()

    def acquire_model(self, manifest: ModelManifest, loader: IModelLoader) -> LoadedModel:
        """Validates available VRAM and safely loads a model onto the GPU."""
        with self._lock:
            # 1. Check if we already have it loaded (Idempotency)
            for active_model in self._active_models.values():
                if active_model.manifest.model_path == manifest.model_path:
                    return active_model

            # 2. The Bouncer Check: Query the hardware directly
            status = self._monitor.get_vram_status()
            required_vram = manifest.required_vram_mb + self.VRAM_SAFETY_BUFFER_MB

            if status.free_vram_mb < required_vram:
                raise exc.OutOfVramError(
                    f"Cannot load {manifest.model_type.value}. "
                    f"Requires {required_vram:.2f} MB (including safety buffer), "
                    f"but only {status.free_vram_mb:.2f} MB is free. "
                    f"Please release active models first."
                )

            # 3. Safe to load
            try:
                loaded_model = loader.load(manifest)
                self._active_models[loaded_model.id] = loaded_model
                return loaded_model

            except Exception as e:
                if isinstance(e, exc.ModelManagerError):
                    raise e
                raise exc.ModelLoadError(f"Unexpected failure loading model: {str(e)}", original_exception=e) from e

    def release_model(self, loaded_model: LoadedModel, loader: IModelLoader, flush: bool = True) -> None:
        """Explicitly unloads a model."""
        with self._lock:
            if loaded_model.id not in self._active_models:
                return  # Already released

            try:
                # 1. Command the feature's adapter to delete the PyTorch instance
                loader.unload(loaded_model)

                # 2. Remove from our tracking state
                del self._active_models[loaded_model.id]

                # 3. Force Aggressive Memory Flush (if requested)
                if flush:
                    self._monitor.flush_memory()

            except Exception as e:
                raise exc.ModelManagerError(f"Failed to release model safely: {str(e)}", original_exception=e) from e

    def release_all(self, loaders_map: Dict[uuid.UUID, IModelLoader]) -> None:
        """Emergency flush. Unloads everything currently tracked."""
        with self._lock:
            active_ids = list(self._active_models.keys())
            for model_id in active_ids:
                if model_id in loaders_map:
                    # Because we use RLock, calling this locked method from within another locked block is now perfectly safe.
                    self.release_model(self._active_models[model_id], loaders_map[model_id], flush=False)

            # Perform ONE massive flush at the very end
            self._monitor.flush_memory()