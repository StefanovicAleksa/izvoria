# Canvas: orchestrator_v6
# Path: app/core/model_manager/service/orchestrator.py

import uuid
import threading
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IVramMonitor, IModelLoader
from app.core.model_manager import exceptions as exc
from app.core.model_manager.utils.error_mapper import map_model_manager_error

logger = logging.getLogger(__name__)


class ModelManagerOrchestrator:
    VRAM_SAFETY_BUFFER_MB = 500.0
    WAIT_POLL_INTERVAL_SEC = 0.1

    def __init__(self, vram_monitor: IVramMonitor):
        self._monitor = vram_monitor
        self._active_models: Dict[uuid.UUID, LoadedModel] = {}
        self._active_loaders: Dict[uuid.UUID, IModelLoader] = {}
        # How many times each resident model has been handed out by acquire_model. Used ONLY by
        # release_preload, to tell an abandoned preload apart from one somebody has since taken.
        # Deliberately not a refcount: `release_after=False` means a holder's acquires intentionally
        # outnumber its releases (a Phase holds one residency across many feature-API calls), so
        # freeing on a balanced count would leak every model the report path loads.
        self._handouts: Dict[uuid.UUID, int] = {}
        self._historical_backpacks: Dict[str, float] = {}
        self._lock = threading.RLock()

    def _find_resident(self, manifest: ModelManifest) -> Optional[LoadedModel]:
        """
        Residency is keyed on (model_type, model_path), not the path alone: the same
        GGUF file is loaded by several adapters with different settings (e.g. n_ctx),
        so a resident instance must never be handed to a different feature.
        Callers must hold `self._lock`.
        """
        for active_model in self._active_models.values():
            if (active_model.manifest.model_type == manifest.model_type
                    and active_model.manifest.model_path == manifest.model_path):
                return active_model
        return None

    def _wait_until_clear(self, manifest: ModelManifest) -> None:
        """
        Waits, without a time limit, until no model is resident (or this manifest's model is). A resident model
        belongs to a phase that is still working, and batch phases run as long as their input needs: a limit
        here failed a 74-minute ingestion whose visual analysis took 174 s. Leaks are prevented where they
        start instead: every pipeline discards unused preloads and the engine releases all models after a run.
        """
        logger.info("Background preloader parking politely until active models clear VRAM...")
        while True:
            with self._lock:
                if not self._active_models or self._find_resident(manifest):
                    return
            time.sleep(self.WAIT_POLL_INTERVAL_SEC)

    def acquire_model(
            self,
            manifest: ModelManifest,
            loader: IModelLoader,
            wait_for_clear: bool = False
    ) -> LoadedModel:
        # 1. Idempotency Check
        with self._lock:
            resident = self._find_resident(manifest)
            if resident:
                return self._hand_out(resident)

        # 2. Prevent Race Conditions: Background preloads must wait for active inference to finish
        if wait_for_clear:
            self._wait_until_clear(manifest)

        with self._lock:
            # Double check idempotency after acquiring lock
            resident = self._find_resident(manifest)
            if resident:
                return self._hand_out(resident)

            # 3. Swap logic: If VRAM is occupied, release existing models
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
                self._handouts[loaded_model.id] = 1

                self._monitor.reset_peak_stats()
                return loaded_model

            except Exception as e:
                self._monitor.flush_memory()
                raise map_model_manager_error(e) from e

    def _hand_out(self, model: LoadedModel) -> LoadedModel:
        """Records that a resident model has been given to one more caller. Callers must hold `self._lock`."""
        self._handouts[model.id] = self._handouts.get(model.id, 0) + 1
        return model

    def release_preload(self, loaded_model: LoadedModel, loader: IModelLoader) -> None:
        """
        Releases a preload nobody resolved -- unless somebody has taken the model in the meantime.

        This is the fire-and-forget release. Every other release is made by the code holding the model,
        which knows it has finished with it; this one is made by a background callback on behalf of a
        caller that walked away, and by the time it runs the model may have been handed to somebody else.

        Freeing it unconditionally is a use-after-free: `acquire_model` returns a resident model straight
        out of the idempotency check, so a phase can be mid-generation on the very instance this callback
        frees. That is an abort inside llama.cpp (`free_model`), not an exception -- the process dies.

        The check and the free both happen under `self._lock`, and `acquire_model` counts its hand-out
        under the same lock, so the two orderings are the only ones possible: either the acquire lands
        first and this becomes a no-op, or this frees first and the acquire loads a fresh copy.
        """
        with self._lock:
            if loaded_model.id not in self._active_models:
                return

            handouts = self._handouts.get(loaded_model.id, 0)
            if handouts > 1:
                # Somebody acquired it after the preload did. Drop only the preload's own claim; the
                # holder's release_model frees it when it is actually finished.
                self._handouts[loaded_model.id] = handouts - 1
                logger.info(
                    f"Abandoned preload of {loaded_model.manifest.model_type.value} is still in use by "
                    f"another caller; leaving it resident."
                )
                return

            self.release_model(loaded_model, loader, flush=True)

    def release_model(self, loaded_model: LoadedModel, loader: IModelLoader, flush: bool = True) -> None:
        with self._lock:
            if loaded_model.id not in self._active_models:
                return

            try:
                loader.unload(loaded_model)
                del self._active_models[loaded_model.id]

                if loaded_model.id in self._active_loaders:
                    del self._active_loaders[loaded_model.id]
                self._handouts.pop(loaded_model.id, None)

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
