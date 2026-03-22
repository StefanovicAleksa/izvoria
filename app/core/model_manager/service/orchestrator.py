# Canvas: orchestrator_v5
# Path: app/core/model_manager/service/orchestrator.py
import uuid
import threading
from pathlib import Path
from typing import Dict

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IVramMonitor, IModelLoader
from app.core.model_manager import exceptions as exc
from app.core.model_manager.utils.error_mapper import map_model_manager_error


class ModelManagerOrchestrator:
    """
    The Active VRAM Air Traffic Controller.
    Auto-discovers weight sizes and dynamically learns activation spikes (backpacks) via telemetry.
    Strictly handles and maps OS and Framework-level exceptions.
    """

    VRAM_SAFETY_BUFFER_MB = 500.0

    def __init__(self, vram_monitor: IVramMonitor):
        self._monitor = vram_monitor
        self._active_models: Dict[uuid.UUID, LoadedModel] = {}

        # Telemetry Memory Bank: model_path -> highest_observed_activation_spike_in_mb
        self._historical_backpacks: Dict[str, float] = {}

        self._lock = threading.RLock()

    def _get_actual_size_mb(self, model_path: str, fallback_mb: float) -> float:
        """Auto-discovers the exact size of the model on disk."""
        try:
            path = Path(model_path)
            if not path.exists():
                return fallback_mb

            # If it's a single file (like .gguf or .nemo)
            if path.is_file():
                return path.stat().st_size / (1024 ** 2)

            # If it's a huggingface directory (like faster-whisper)
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total_size / (1024 ** 2)
        except OSError:
            # Safely catch file system bounds (PermissionError, FileNotFoundError)
            # and gracefully fall back to the config manifest size
            return fallback_mb

    def record_telemetry(self) -> None:
        """Called constantly by the Background Worker to learn model spikes."""
        try:
            current_alloc, peak_alloc = self._monitor.get_pytorch_memory_stats()
            backpack_mb = peak_alloc - current_alloc

            # If tensors have spiked above baseline, attribute the spike to the active models
            if backpack_mb > 0:
                with self._lock:
                    for active_model in self._active_models.values():
                        path = active_model.manifest.model_path
                        existing = self._historical_backpacks.get(path, 0.0)

                        # Only update if we hit a new high-water mark
                        if backpack_mb > existing:
                            self._historical_backpacks[path] = backpack_mb
        except (RuntimeError, ValueError, AttributeError):
            # The telemetry daemon should silently ignore tracking failures
            # (e.g., if the CUDA device temporarily resets)
            pass

    def acquire_model(self, manifest: ModelManifest, loader: IModelLoader) -> LoadedModel:
        """Validates available VRAM dynamically and safely loads a model onto the GPU."""
        with self._lock:
            # 1. Check if already loaded
            for active_model in self._active_models.values():
                if active_model.manifest.model_path == manifest.model_path:
                    return active_model

            # 2. Dynamic Requirement Calculation
            actual_size_mb = self._get_actual_size_mb(manifest.model_path, manifest.required_vram_mb)
            learned_backpack = self._historical_backpacks.get(manifest.model_path, 0.0)

            required_vram = actual_size_mb + learned_backpack + self.VRAM_SAFETY_BUFFER_MB
            status = self._monitor.get_vram_status()

            # 3. The Bouncer Check
            if status.free_vram_mb < required_vram:
                raise exc.OutOfVramError(
                    f"Cannot load {manifest.model_type.value}.\n"
                    f"Needs {required_vram:.2f} MB (Weights: {actual_size_mb:.2f} | Learned Spike: {learned_backpack:.2f} | Buffer: {self.VRAM_SAFETY_BUFFER_MB}), "
                    f"but only {status.free_vram_mb:.2f} MB is free.\n"
                    f"Please release active models first."
                )

            # 4. Safe to load
            try:
                loaded_model = loader.load(manifest)
                self._active_models[loaded_model.id] = loaded_model

                # Reset peak stats so the telemetry daemon measures from this new baseline
                self._monitor.reset_peak_stats()
                return loaded_model

            except (RuntimeError, MemoryError, OSError, ValueError, EOFError) as e:
                # Intercept underlying PyTorch/C++ framework errors and translate them cleanly
                raise map_model_manager_error(e) from e
            except exc.ModelManagerError as e:
                # If the adapter already threw a domain error, let it pass through
                raise e

    def release_model(self, loaded_model: LoadedModel, loader: IModelLoader, flush: bool = True) -> None:
        """Explicitly unloads a model."""
        with self._lock:
            if loaded_model.id not in self._active_models:
                return

            try:
                loader.unload(loaded_model)
                del self._active_models[loaded_model.id]

                if flush:
                    self._monitor.flush_memory()

                self._monitor.reset_peak_stats()

            except (RuntimeError, MemoryError, OSError, AttributeError) as e:
                raise map_model_manager_error(e) from e
            except exc.ModelManagerError as e:
                raise e

    def release_all(self, loaders_map: Dict[uuid.UUID, IModelLoader]) -> None:
        """Emergency flush."""
        with self._lock:
            active_ids = list(self._active_models.keys())
            for model_id in active_ids:
                if model_id in loaders_map:
                    self.release_model(self._active_models[model_id], loaders_map[model_id], flush=False)

            self._monitor.flush_memory()
            self._monitor.reset_peak_stats()