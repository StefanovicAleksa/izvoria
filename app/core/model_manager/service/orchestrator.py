# Canvas: orchestrator_v6
# Path: app/core/model_manager/service/orchestrator.py

import itertools
import uuid
import threading
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.domain.interfaces import IVramMonitor, IModelLoader
from app.core.model_manager import exceptions as exc
from app.core.model_manager.utils.error_mapper import map_model_manager_error

logger = logging.getLogger(__name__)


class ModelManagerOrchestrator:
    """
    The registry every GPU model is loaded and released through.

    A resident model is ACTIVE from `acquire_model` until its holder's `release_model`. A model whose type is in
    `keep_warm_types` then becomes IDLE instead of being unloaded: it stays in VRAM, is handed out again for free,
    and is evicted only when a later load needs the room (the fewest megabytes that make it, least recently
    used on a tie). Every other model is unloaded
    on release, as before. Without `keep_warm_types` nothing is ever IDLE and the pool is the single slot it always
    was: every acquire evicts the other residents, and a waiting acquire waits for an empty GPU.

    Two further kinds of type exist for the hosted, multi-user API (both empty on the appliance):
    - `pinned_types` are loaded once (`warm_up`) and then never released, evicted or unloaded, until
      `shutdown_all`. Concurrent requests share them safely because nothing can free them mid-use.
    - `external_types` are served by another process (llama-server): `acquire_model` hands out one client per
      (type, path) without budgeting VRAM for it, and releasing it does nothing.
    With `strict_resident`, acquiring any other type raises ResidencyError: with requests running concurrently, a
    load or release outside those two kinds could free a model another request is using (release_model is not a
    refcount), so a concurrent engine must never reach that lifecycle.
    """
    VRAM_SAFETY_BUFFER_MB = 500.0
    WAIT_POLL_INTERVAL_SEC = 0.1
    # Above this many IDLE models the cheapest-eviction search gives way to evicting them oldest first
    MAX_EVICTION_CANDIDATES = 10

    def __init__(self, vram_monitor: IVramMonitor, keep_warm_types: Optional[Iterable[ModelType]] = None,
                 pinned_types: Optional[Iterable[ModelType]] = None,
                 external_types: Optional[Iterable[ModelType]] = None,
                 strict_resident: bool = False):
        self._monitor = vram_monitor
        self._keep_warm_types = frozenset(keep_warm_types or ())
        self._pinned_types = frozenset(pinned_types or ())
        self._external_types = frozenset(external_types or ())
        overlap = self._pinned_types & self._external_types
        if overlap:
            raise ValueError(f"Model types cannot be both pinned and external: {sorted(t.value for t in overlap)}")
        self._strict_resident = strict_resident
        # Models released by their holder but kept resident, with when they went idle (for LRU eviction).
        # A resident model not in here is ACTIVE: somebody holds it, and it is never evicted by a waiting acquire.
        self._idle_since: Dict[uuid.UUID, float] = {}
        # VRAM a load actually took (free before minus free after), per (model_type, model_path). On-disk size is
        # a poor guess for both directions: models/bge-m3 is 4.3 GB on disk and 1.1 GB loaded, while llama.cpp
        # adds its KV cache and compute buffers on top of the GGUF. Learned per process, like the backpack.
        self._footprints: Dict[Tuple[ModelType, str], float] = {}
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

        With models kept warm it returns as soon as the model would fit beside the ACTIVE ones once IDLE models are
        evicted, so a preload loads while the current stage is still computing instead of after it.
        """
        logger.info("Background preloader parking politely until active models clear VRAM...")
        while True:
            with self._lock:
                if not self._held_ids() or self._find_resident(manifest):
                    return
                if self._keep_warm_types and self._could_fit_beside_held(manifest):
                    return
            time.sleep(self.WAIT_POLL_INTERVAL_SEC)

    def _is_permanent(self, model_type: ModelType) -> bool:
        """Pinned or external: never released, evicted or unloaded short of `shutdown_all`."""
        return model_type in self._pinned_types or model_type in self._external_types

    def _held_ids(self) -> list:
        """
        Resident models somebody holds (ACTIVE). Pinned and external models are not counted: they are held by
        nobody in particular and never released, so a waiting acquire must not wait for them.
        Callers must hold `self._lock`.
        """
        return [mid for mid, model in self._active_models.items()
                if mid not in self._idle_since and not self._is_permanent(model.manifest.model_type)]

    def _could_fit_beside_held(self, manifest: ModelManifest) -> bool:
        """A cheap check, without flushing: free VRAM plus what evicting every IDLE model would return."""
        reclaimable = sum(self._footprint_mb(self._active_models[mid].manifest) for mid in self._idle_since)
        free = self._monitor.get_vram_status().free_vram_mb + reclaimable
        return free >= self._required_vram_mb(manifest, alongside=True)

    def acquire_model(
            self,
            manifest: ModelManifest,
            loader: IModelLoader,
            wait_for_clear: bool = False
    ) -> LoadedModel:
        """
        `wait_for_clear=False` (ingestion and batch APIs) evicts every other resident, ACTIVE or not: a batch phase
        owns the GPU. `wait_for_clear=True` (live APIs, every preload) never evicts an ACTIVE model; it evicts IDLE
        ones (the fewest megabytes that make room), and waits for ACTIVE ones to be released if that is not enough room.
        Pinned and external models are never evicted by either.
        """
        if manifest.model_type in self._external_types:
            return self._acquire_external(manifest, loader)
        if self._strict_resident and manifest.model_type not in self._pinned_types:
            raise exc.ResidencyError(
                f"{manifest.model_type.value} is neither pinned nor external, so it can't be loaded while requests "
                f"run concurrently. Add it to MODEL_PINNED_TYPES or MODEL_EXTERNAL_TYPES."
            )

        while True:
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

                if wait_for_clear and not self._keep_warm_types and self._held_ids():
                    # Single-slot pool: somebody took the GPU between the wait and the lock; wait for it again
                    resident = None
                else:
                    resident = self._admit(manifest, loader, wait_for_clear)
                if resident:
                    return resident

            time.sleep(self.WAIT_POLL_INTERVAL_SEC)

    def _acquire_external(self, manifest: ModelManifest, loader: IModelLoader) -> LoadedModel:
        """One client per (type, path), created on first use. Nothing is budgeted: the VRAM is another process's."""
        with self._lock:
            resident = self._find_resident(manifest)
            if resident:
                return self._hand_out(resident)
            try:
                logger.info(f"Connecting {manifest.model_type.value} to its external server")
                loaded_model = loader.load(manifest)
            except Exception as e:
                raise map_model_manager_error(e) from e
            self._active_models[loaded_model.id] = loaded_model
            self._active_loaders[loaded_model.id] = loader
            self._handouts[loaded_model.id] = 1
            return loaded_model

    def warm_up(self, manifest: ModelManifest, loader: IModelLoader) -> LoadedModel:
        """
        Loads a pinned (or connects an external) model ahead of the first request, so no request ever waits on a load.

        A pinned model is admitted against its declared `required_vram_mb`, not its on-disk size: it loads once, at
        startup, beside an external server that already holds its share of the card, and on-disk size misjudges it
        badly (models/bge-m3 is 4.3 GB on disk, 1.1 GB loaded; that estimate refused the embedder next to
        llama-server on a 12 GB card with 4.6 GB free). `<FEATURE>_VRAM_REQUIREMENT_MB` is the declared upper bound.
        """
        if not self._is_permanent(manifest.model_type):
            raise ValueError(f"{manifest.model_type.value} is not pinned or external; warm_up would leak it")
        if manifest.model_type in self._external_types:
            return self._acquire_external(manifest, loader)
        with self._lock:
            resident = self._find_resident(manifest)
            if resident:
                return self._hand_out(resident)
            self._monitor.flush_memory()
            status = self._monitor.get_vram_status()
            needed = manifest.required_vram_mb + self.VRAM_SAFETY_BUFFER_MB
            if status.free_vram_mb < needed:
                raise exc.OutOfVramError(
                    f"Insufficient VRAM to pin {manifest.model_type.value}.\n"
                    f"Needs: {needed:.1f}MB | Free: {status.free_vram_mb:.1f}MB"
                )
            return self._load(manifest, loader, free_before_mb=status.free_vram_mb)

    def _admit(self, manifest: ModelManifest, loader: IModelLoader, wait_for_clear: bool) -> Optional[LoadedModel]:
        """
        Makes room, then loads; None means wait (the room will come from ACTIVE models being released).
        Callers must hold `self._lock`.
        """
        status, alongside = self._make_room(manifest, evict_held=not wait_for_clear)
        required_vram = self._required_vram_mb(manifest, alongside=alongside)

        if status.free_vram_mb >= required_vram:
            return self._load(manifest, loader, free_before_mb=status.free_vram_mb)

        if wait_for_clear and self._held_ids():
            # The cheap estimate said it would fit but the flushed numbers disagree; the ACTIVE
            # models' releases will make the room, so wait for them rather than fail
            logger.info(f"{manifest.model_type.value} does not fit beside the models in use yet; waiting.")
            return None

        raise exc.OutOfVramError(
            f"Insufficient VRAM for {manifest.model_type.value}.\n"
            f"Needs: {required_vram:.1f}MB | Free: {status.free_vram_mb:.1f}MB"
        )

    def _make_room(self, manifest: ModelManifest, evict_held: bool):
        """
        Evicts what this load may evict and returns (VRAM status after a deep flush, whether models stay resident).
        Only IDLE models are evicted, and only as many megabytes as the load needs. Callers must hold `self._lock`.
        """
        # Swap logic: a batch acquire releases every current resident (pinned and external ones stay)
        evictable = [mid for mid, model in self._active_models.items()
                     if not self._is_permanent(model.manifest.model_type)]
        if evict_held and evictable:
            logger.info("Swapping models. Releasing current residents...")
            for mid in evictable:
                self._unload_resident(mid, flush=False)

        # PRE-EMPTIVE DEEP FLUSH
        self._monitor.flush_memory()
        time.sleep(0.1)
        status = self._monitor.get_vram_status()

        while self._active_models:
            shortfall = self._required_vram_mb(manifest, alongside=True) - status.free_vram_mb
            if shortfall <= 0 or not self._idle_since:
                return status, True
            for victim in self._cheapest_eviction(shortfall):
                logger.info(
                    f"Evicting idle {self._active_models[victim].manifest.model_type.value} "
                    f"to make room for {manifest.model_type.value} "
                    f"(short {shortfall:.0f} MB; free {status.free_vram_mb:.0f} MB, "
                    f"footprint {self._footprint_mb(manifest):.0f} MB, backpacks {self._backpack_summary()})."
                )
                self._unload_resident(victim, flush=True)
            status = self._monitor.get_vram_status()

        return status, False

    def _backpack_summary(self) -> str:
        return ", ".join(f"{Path(path).name}={mb:.0f}" for path, mb in self._historical_backpacks.items()) or "none"

    def _cheapest_eviction(self, shortfall_mb: float) -> list:
        """
        The IDLE models to evict for `shortfall_mb`: the set that frees the fewest megabytes while making the room,
        least recently used on a tie, so a 1 GB reranker is never made room for by evicting the 7.8 GB answer model
        when the 1 GB embedder would do. If no set is enough, all of them. Callers must hold `self._lock`.
        """
        idle = sorted(self._idle_since, key=self._idle_since.get)
        sizes = {mid: self._footprint_mb(self._active_models[mid].manifest) for mid in idle}
        if len(idle) > self.MAX_EVICTION_CANDIDATES:
            return idle
        best = None
        for count in range(1, len(idle) + 1):
            for subset in itertools.combinations(idle, count):
                freed = sum(sizes[mid] for mid in subset)
                if freed < shortfall_mb:
                    continue
                key = (freed, max(self._idle_since[mid] for mid in subset))
                if best is None or key < best[0]:
                    best = (key, list(subset))
        return best[1] if best else idle

    def _load(self, manifest: ModelManifest, loader: IModelLoader, free_before_mb: float) -> LoadedModel:
        """Safe to load. Callers must hold `self._lock`."""
        try:
            logger.info(f"Loading {manifest.model_type.value}: {manifest.model_path}")
            loaded_model = loader.load(manifest)
            self._active_models[loaded_model.id] = loaded_model
            self._active_loaders[loaded_model.id] = loader
            self._handouts[loaded_model.id] = 1
            self._learn_footprint(manifest, free_before_mb)

            self._monitor.reset_peak_stats()
            return loaded_model

        except Exception as e:
            self._monitor.flush_memory()
            raise map_model_manager_error(e) from e

    def _learn_footprint(self, manifest: ModelManifest, free_before_mb: float) -> None:
        try:
            taken = free_before_mb - self._monitor.get_vram_status().free_vram_mb
        except Exception:
            return
        if taken > 0:
            key = (manifest.model_type, manifest.model_path)
            # The larger of every measurement: a stage computing on another thread can free memory mid-load
            self._footprints[key] = max(self._footprints.get(key, 0.0), taken)

    def _footprint_mb(self, manifest: ModelManifest) -> float:
        """What the model takes in VRAM: measured if it has been loaded before, else the larger estimate."""
        learned = self._footprints.get((manifest.model_type, manifest.model_path))
        if learned is not None:
            return learned
        return max(self._get_actual_size_mb(manifest.model_path, manifest.required_vram_mb), manifest.required_vram_mb)

    def _required_vram_mb(self, manifest: ModelManifest, alongside: bool) -> float:
        """
        Alone on the GPU: on-disk size + learned backpack + buffer, unchanged since the single-slot pool, so
        ingestion is admitted exactly as before. Beside other residents: the model's footprint plus room for the
        largest activation spike among it and them (one model computes at a time) plus the buffer.
        """
        learned_backpack = self._historical_backpacks.get(manifest.model_path, 0.0)
        if not alongside:
            actual_size_mb = self._get_actual_size_mb(manifest.model_path, manifest.required_vram_mb)
            return actual_size_mb + learned_backpack + self.VRAM_SAFETY_BUFFER_MB
        spike = max(
            [learned_backpack] + [self._historical_backpacks.get(m.manifest.model_path, 0.0)
                                  for m in self._active_models.values()]
        )
        return self._footprint_mb(manifest) + spike + self.VRAM_SAFETY_BUFFER_MB

    def _hand_out(self, model: LoadedModel) -> LoadedModel:
        """Records that a resident model has been given to one more caller. Callers must hold `self._lock`."""
        self._idle_since.pop(model.id, None)
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
            if loaded_model.id not in self._active_models or self._is_permanent(loaded_model.manifest.model_type):
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
        """
        The holder is done with the model. A keep-warm type goes IDLE (stays resident, evictable); anything else is
        unloaded. Releasing a model that is already IDLE or gone, or a pinned or external one, is a no-op.
        """
        with self._lock:
            if loaded_model.id not in self._active_models or loaded_model.id in self._idle_since:
                return
            if self._is_permanent(loaded_model.manifest.model_type):
                return

            if loaded_model.manifest.model_type in self._keep_warm_types:
                self._idle_since[loaded_model.id] = time.monotonic()
                self._handouts[loaded_model.id] = 0
                if flush:
                    # Hands the activations the model cached back to the device; the weights stay
                    self._monitor.flush_memory()
                self._monitor.reset_peak_stats()
                logger.info(f"Keeping {loaded_model.manifest.model_type.value} warm in VRAM")
                return

            self._unload(loaded_model, loader, flush=flush)

    def _unload_resident(self, model_id: uuid.UUID, flush: bool, fallback_loader: Optional[IModelLoader] = None) -> None:
        """Truly unloads a resident model, IDLE or ACTIVE. Callers must hold `self._lock`."""
        loader = self._active_loaders.get(model_id) or fallback_loader
        if loader:
            self._unload(self._active_models[model_id], loader, flush=flush)

    def _unload(self, loaded_model: LoadedModel, loader: IModelLoader, flush: bool) -> None:
        with self._lock:
            if loaded_model.id not in self._active_models:
                return

            try:
                loader.unload(loaded_model)
                del self._active_models[loaded_model.id]

                if loaded_model.id in self._active_loaders:
                    del self._active_loaders[loaded_model.id]
                self._handouts.pop(loaded_model.id, None)
                self._idle_since.pop(loaded_model.id, None)

                if flush:
                    self._monitor.flush_memory()

                self._monitor.reset_peak_stats()
                logger.info(f"Successfully released {loaded_model.manifest.model_type.value}")

            except Exception as e:
                raise map_model_manager_error(e) from e

    def release_all(self, loaders_map: Dict[uuid.UUID, IModelLoader]) -> None:
        """
        Truly unloads every resident model, IDLE ones included: this is what frees the GPU for ingestion. Pinned
        and external models stay, and with nothing else resident the CUDA cache is left alone, so a pinned model's
        reserved activation memory is not handed to another process between requests.
        """
        with self._lock:
            unloaded = False
            for model_id, model in list(self._active_models.items()):
                if self._is_permanent(model.manifest.model_type):
                    continue
                self._unload_resident(model_id, flush=False, fallback_loader=loaders_map.get(model_id))
                unloaded = True

            if unloaded or not (self._pinned_types or self._external_types):
                self._monitor.flush_memory()
                self._monitor.reset_peak_stats()
            logger.info("Emergency release_all completed. VRAM is clear.")

    def shutdown_all(self, loaders_map: Dict[uuid.UUID, IModelLoader]) -> None:
        """Unloads everything, pinned and external models included. Only for process shutdown."""
        with self._lock:
            for model_id in list(self._active_models.keys()):
                self._unload_resident(model_id, flush=False, fallback_loader=loaders_map.get(model_id))
            self._monitor.flush_memory()
            self._monitor.reset_peak_stats()

    def resident_types(self) -> list:
        """The types currently resident, for health reporting."""
        with self._lock:
            return [model.manifest.model_type for model in self._active_models.values()]

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
        """
        Learns the backpack of the one model in use. The counters are read under the lock, because loads and
        unloads hold it: read outside, a sample taken mid-eviction (a model already freed, the peak not yet reset)
        reads the freed model's size as a spike, and was charged to every warm model -- 1084 MB, BGE-M3's size, on
        the embedder and on llama.cpp's Qwen, which allocates nothing through torch. With two models in use the
        spike can't be told apart, so nothing is learned.
        """
        try:
            with self._lock:
                held = self._held_ids()
                if len(held) != 1:
                    return
                current_alloc, peak_alloc = self._monitor.get_pytorch_memory_stats()
                backpack_mb = peak_alloc - current_alloc
                if backpack_mb > 0:
                    path = self._active_models[held[0]].manifest.model_path
                    if backpack_mb > self._historical_backpacks.get(path, 0.0):
                        self._historical_backpacks[path] = backpack_mb
        except Exception:
            pass

    def get_telemetry(self) -> Dict[str, float]:
        return self._historical_backpacks
