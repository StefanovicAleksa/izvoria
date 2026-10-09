# model_manager

`app/core/model_manager/`

## What it does

The VRAM-budget-driven traffic controller for every GPU-bound model in the system. Because dev hardware has only 12GB VRAM and no stage of the pipeline can assume its model is the only one that will ever need to load, every model acquisition and release in the codebase goes through this module. It is the single component that makes the whole "load a model, use it, unload it, load the next one" architecture (see the root `CLAUDE.md`) actually work.

## How it does it

- **`domain/entities.py`**: `ModelManifest` (model type, disk path, `required_vram_mb` — a per-model estimate sourced from `app/core/config.py`), `LoadedModel` (uuid + manifest + an opaque `model_instance: Any`), `VramStatus`.
- **`domain/interfaces.py`**: `IVramMonitor` (status/flush/pytorch-stats/reset-peak) and `IModelLoader` (`load`/`unload`) — every feature's model adapter implements `IModelLoader`, which is what lets the orchestrator treat wildly different model types (NeMo, faster-whisper, HF Transformers, llama.cpp) uniformly.
- **`service/orchestrator.py` — `ModelManagerOrchestrator(monitor, keep_warm_types=...)`**: the actual registry. `OnyxBootstrapper` passes `MODEL_KEEP_WARM_TYPES` (embedder, reranker, `QA_GENERATION`, `REPORT_AUTHORING`) unless `MODEL_KEEP_WARM=False`; without it the pool is single-slot. Holds `_active_models: Dict[uuid, LoadedModel]` and `_active_loaders: Dict[uuid, IModelLoader]`, guarded by an `RLock`.
  - `acquire_model(manifest, loader, wait_for_clear=False)`: idempotency check under the lock (same `(model_type, model_path)` already resident, ACTIVE or IDLE → returns it, now ACTIVE; the same weights loaded by a *different* feature are never shared, because adapters load them with different settings such as `n_ctx`) → makes room → deep VRAM flush → admission control against `IVramMonitor.get_vram_status().free_vram_mb` (raises `OutOfVramError` if insufficient) → `loader.load(manifest)`, recording the VRAM the load actually took.
    - `wait_for_clear=False` (ingestion and batch APIs) **evicts every other resident**, ACTIVE or not.
    - `wait_for_clear=True` (live APIs and every preload) never evicts an ACTIVE model. It waits, with no time limit, until the model fits beside the ACTIVE ones once IDLE ones are evicted (the set that frees the fewest megabytes while making room, least recently used on a tie). A resident model belongs to a phase that is still working, and batch phases run as long as their input needs; leaks are prevented by discarding unused preloads and releasing all models after a run, not by giving up. Without keep-warm types it waits for an empty GPU, as the single-slot pool did.
    - Required VRAM, alone on the GPU: on-disk size + learned backpack + 500 MB (unchanged, so ingestion is admitted as before). Beside other residents: the model's measured footprint (falling back to `max(on-disk size, required_vram_mb)` before its first load) + the largest backpack among it and the residents + 500 MB.
  - `release_model(loaded_model, loader, flush=True)`: ends the holding. A type in `keep_warm_types` goes IDLE (stays resident, LRU-stamped); anything else is unloaded. Releasing an IDLE or gone model is a no-op.
  - `release_all()`: full purge, IDLE models included. `OnyxEngine` calls it (via `AsyncModelService.release_all`) after every ingestion run and after any failed live call.
  - `record_telemetry()`/`get_telemetry()`: a background thread samples `torch.cuda.memory_allocated()` vs `max_memory_allocated()` every 0.5s and remembers the largest observed "backpack" per model path. It reads the counters under the orchestrator lock (never mid-load or mid-eviction) and records only while exactly one model is ACTIVE, since a shared peak cannot be attributed, refining future admission-control estimates over time.
- **`utils/background_worker.py` — `BackgroundModelWorker`**: a single-thread `ThreadPoolExecutor(max_workers=1)` used to preload models asynchronously while other (I/O-bound) work proceeds; also owns the telemetry thread. `discard_preload(future, loader)` abandons a preload nobody will resolve: a queued task is cancelled, a running or finished one has its model released as soon as the load completes.
- **`service/async_model_service.py` — `AsyncModelService`**: a thin facade — `trigger_preload()` returns a `Future`, `resolve_preload()` blocks on it, `trigger_unload()` is deliberately **synchronous** (bypasses the worker pool) so it can't deadlock behind a parked preload task. `discard_preload()` and `release_all()` delegate to the worker.
- **`utils/torchcodec_guard.py` — `block_torchcodec_import()`**: the shared context manager every sentence-transformers adapter (both embedders, the reranker) wraps its model import in, temporarily poisoning `sys.modules["torchcodec"]` so torchvision doesn't crash on torchcodec's broken bindings, and restoring it on exit.
- **`data/vram_monitor.py` — `PyTorchVramMonitor`**: talks to `torch.cuda.mem_get_info()` for real hardware totals, and performs the "aggressive flush" sequence: `gc.collect()` → `torch.cuda.synchronize()` → `empty_cache()` → `ipc_collect()` → `gc.collect()` again.

## Inputs / Outputs

Input: a `ModelManifest` + an `IModelLoader` implementation. Output: a `LoadedModel` handle (on `acquire_model`) whose `model_instance` the caller uses directly, then hands back to `release_model` when done.

## Where it fits

Not a pipeline stage itself — every GPU-bound Phase (VAD, transcription, diarization, domain adaptation, phonetic correction, contextual retrieval, all five visual features, RAG embedding, retrieval scrubbing, QA generation) acquires and releases its model through this orchestrator. `OnyxBootstrapper.build()` constructs exactly one `ModelManagerOrchestrator` shared across the whole application.

## Gotchas / design notes

- **Warm pool.** Measured on the 12 GB dev card (2026-10-03): BGE-M3 1090 MB, bge-reranker 1088 MB, Qwen3.5-9B at 128K q8_0 7788 MB. All three stay resident with ~1.1 GB free, so a warm chat turn loads nothing. The Qwen2.5-7B scrubber does not fit beside them, so `RetrievalScrubberAPI` loads it only when a chunk scores below `RETRIEVAL_SCRUBBER_TRUST_RERANK_SCORE`, and the live pipelines no longer preload it. The first turn after start may evict the embedder/reranker to fit the answer model's conservative first estimate (10000 MB); once measured, all three fit.
- **The follow-up rewrite borrows the answer model** (`ReasoningAPI(rewrite_manifest=, rewrite_loader=)`), loaded by the QA adapter so it keeps its settings. The Reasoned-Answer loop keeps `QA_REASONER`: two phases holding one instance would let one's release free it under the other.
- IDLE models are evicted only for room, never on a timer. `OnyxEngine` releases everything when ingestion starts and ends, after a report, after a failed live call, and on `stop()` (which a language switch calls).
- VRAM budgets in `config.py` are *starting* estimates; the telemetry-refined "backpack" number grows more accurate the longer the process runs, so a freshly-added model's first few loads use a more conservative (safer, more likely to under-utilize available VRAM) estimate.
- No multi-GPU/device-affinity concept exists anywhere in this module or in `config.py` — no `device_id`/`CUDA_VISIBLE_DEVICES`. This is the main gap for the target 2×RTX 3060 client hardware (see root `CLAUDE.md`).

## See also

`.claude/skills/onyx-model-lifecycle/SKILL.md` — step-by-step rules for writing a model adapter, feature API, Phase and pipeline integration against this module, with the matching unit-test patterns.
