# Canvas: orchestrator_v5
# Path: app/core/model_manager/service/orchestrator.py

import uuid
import threading
import logging
import time
from pathlib import Path
from typing import Dict

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IVramMonitor, IModelLoader
from app.core.model_manager import exceptions as exc
from app.core.model_manager.utils.error_mapper import map_model_manager_error

logger = logging.getLogger(__name__)


class ModelManagerOrchestrator:
    VRAM_SAFETY_BUFFER_MB = 500.0

    def __init__(self, vram_monitor: IVramMonitor):
        self._monitor = vram_monitor
        self._active_models: Dict[uuid.UUID, LoadedModel] = {}
        self._active_loaders: Dict[uuid.UUID, IModelLoader] = {}
        self._historical_backpacks: Dict[str, float] = {}
        self._lock = threading.RLock()

    def acquire_model(self, manifest: ModelManifest, loader: IModelLoader, wait_for_clear: bool = False) -> LoadedModel:
        # 1. Idempotency Check (Unprotected read is safe here for quick exit)
        for active_model in self._active_models.values():
            if active_model.manifest.model_path == manifest.model_path:
                return active_model

        # 2. Prevent Race Conditions: Background preloads must wait for active inference to finish
        if wait_for_clear:
            logger.info(f"Background preloader parking politely until active models clear VRAM...")
            while self._active_models:
                time.sleep(0.1)

        with self._lock:
            # Double check idempotency after acquiring lock
            for active_model in self._active_models.values():
                if active_model.manifest.model_path == manifest.model_path:
                    return active_model

            # 3. Swap logic: If VRAM is occupied (and we are on the main thread), release existing models
            if self._active_models:
                logger.info("Swapping models. Releasing current residents...")
                active_ids = list(self._active_models.keys())
                for mid in active_ids:
                    correct_loader = self._active_loaders.get(mid)
                    if correct_loader:
                        self.release_model(self._active_models[mid], correct_loader, flush=False)

            # 4. PRE-EMPTIVE DEEP FLUSH
            self._monitor.flush_memory()
            time.sleep(0.1)

            # 5. Calculate dynamic requirement
            actual_size_mb = self._get_actual_size_mb(manifest.model_path, manifest.required_vram_mb)
            learned_backpack = self._historical_backpacks.get(manifest.model_path, 0.0)
            required_vram = actual_size_mb + learned_backpack + self.VRAM_SAFETY_BUFFER_MB

            status = self._monitor.get_vram_status()

            # 6. Admission Control
            if status.free_vram_mb < required_vram:
                raise exc.OutOfVramError(
                    f"Insufficient VRAM for {manifest.model_type.value}.\n"
                    f"Needs: {required_vram:.1f}MB | Free: {status.free_vram_mb:.1f}MB"
                )

            # 7. Safe to load
            try:
                logger.info(f"Loading {manifest.model_type.value}: {manifest.model_path}")
                loaded_model = loader.load(manifest)
                self._active_models[loaded_model.id] = loaded_model
                self._active_loaders[loaded_model.id] = loader

                self._monitor.reset_peak_stats()
                return loaded_model

            except Exception as e:
                self._monitor.flush_memory()
                raise map_model_manager_error(e) from e

    def release_model(self, loaded_model: LoadedModel, loader: IModelLoader, flush: bool = True) -> None:
        with self._lock:
            if loaded_model.id not in self._active_models:
                return

            try:
                loader.unload(loaded_model)
                del self._active_models[loaded_model.id]

                if loaded_model.id in self._active_loaders:
                    del self._active_loaders[loaded_model.id]

                if flush:
                    self._monitor.flush_memory()

                self._monitor.reset_peak_stats()
                logger.info(f"Successfully released {loaded_model.manifest.model_type.value}")

            except Exception as e:
                raise map_model_manager_error(e) from e

    def release_all(self, loaders_map: Dict[uuid.UUID, IModelLoader]) -> None:
        with self._lock:
            active_ids = list(self._active_models.keys())
            for model_id in active_ids:
                correct_loader = self._active_loaders.get(model_id) or loaders_map.get(model_id)
                if correct_loader:
                    self.release_model(self._active_models[model_id], correct_loader, flush=False)

            self._monitor.flush_memory()
            self._monitor.reset_peak_stats()
            logger.info("Emergency release_all completed. VRAM is clear.")

    def _get_actual_size_mb(self, model_path: str, fallback_mb: float) -> float:
        try:
            path = Path(model_path)
            if not path.exists():
                return fallback_mb
            if path.is_file():
                return path.stat().st_size / (1024 ** 2)
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total_size / (1024 ** 2)
        except OSError:
            return fallback_mb

    def record_telemetry(self) -> None:
        try:
            current_alloc, peak_alloc = self._monitor.get_pytorch_memory_stats()
            backpack_mb = peak_alloc - current_alloc

            if backpack_mb > 0:
                with self._lock:
                    for active_model in self._active_models.values():
                        path = active_model.manifest.model_path
                        existing = self._historical_backpacks.get(path, 0.0)
                        if backpack_mb > existing:
                            self._historical_backpacks[path] = backpack_mb
        except Exception:
            pass

    def get_telemetry(self) -> Dict[str, float]:
        return self._historical_backpacks