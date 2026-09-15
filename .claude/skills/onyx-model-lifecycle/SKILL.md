---
name: onyx-model-lifecycle
description: How Onyx loads, uses, releases, preloads and reloads GPU models through app/core/model_manager, and the exact patterns a new model adapter, feature service API, orchestration Phase, pipeline and their tests must follow. Use whenever adding or changing code that loads a model, touches VRAM, adds a ModelType, or adds/changes a Phase with preload_model/discard_preload.
---

# Onyx model lifecycle: load, use, release, reload

Onyx runs a dozen GPU models on a 12 GB card, so **no model may assume it stays in VRAM**. Every model is loaded just in time, used, and released, and every load and release goes through `app/core/model_manager/`. This skill describes how that machinery actually behaves (verified against the code) and the patterns each layer must follow. The best reference implementation is the reranker:

| Layer | Reference file |
|---|---|
| Adapter (`IModelLoader`) | `app/features/rag_reranking/data/cross_encoder_adapter.py` |
| Feature service API | `app/features/rag_reranking/service/rerank_api.py` |
| Orchestration Phase | `app/orchestration/service/phases/rag_reranking_phase.py` |
| Pipeline hooks | `app/orchestration/service/pipelines/live_qa_pipeline.py` |
| Wiring | `app/engine/service/bootstrap.py` (`CrossEncoderRerankerAdapter`, `RerankAPI`, `RagRerankingPhase`) |
| Unit tests | `tests/features/rag_reranking/service/test_rerank_api.py`, `tests/orchestration/phases/test_rag_reranking_phase_unit.py` |
| Integration test | `integration_tests/features/rag_reranking/test_rerank_api_integration.py` |

Read the matching reference file before writing the new one, and copy its shape.

## 1. The moving parts

| Piece | File | Role |
|---|---|---|
| `ModelManifest` | `domain/entities.py` | Frozen: `model_type`, `model_path`, `required_vram_mb`. Identifies what to load. |
| `LoadedModel` | `domain/entities.py` | Handle for a resident model: `id`, `manifest`, `model_instance: Any` (the raw library object). |
| `IModelLoader` | `domain/interfaces.py` | `load(manifest) -> LoadedModel`, `unload(model) -> bool`. Every model adapter implements it. |
| `ModelManagerOrchestrator` | `service/orchestrator.py` | The registry and the only thing that calls `load`/`unload`. Exactly one per process. |
| `BackgroundModelWorker` | `utils/background_worker.py` | **One** worker thread for background preloads, plus the VRAM telemetry thread. |
| `AsyncModelService` | `service/async_model_service.py` | What Phases use: `trigger_preload`, `resolve_preload`, `trigger_unload`, `discard_preload`, `release_all`. |
| `PyTorchVramMonitor` | `data/vram_monitor.py` | Free/total VRAM and the deep flush (`gc` → `synchronize` → `empty_cache` → `ipc_collect` → `gc`). |
| `block_torchcodec_import()` | `utils/torchcodec_guard.py` | Context manager required around sentence-transformers imports/construction. |

`ModelType` lives in `domain/enums.py`. Model exceptions live in `exceptions/`: `ModelManagerError` is the base, with `OutOfVramError`, `CudaCrashError`, `ModelNotFoundError` and `ModelLoadError` beneath it.

## 2. What actually happens

### `acquire_model(manifest, loader, wait_for_clear=False)`
1. **Residency check.** If a model with the same **`(model_type, model_path)`** is resident, it is returned as-is and nothing is loaded. The same weights file under a different `ModelType` is *not* shared: adapters load identical GGUFs with different settings (e.g. `n_ctx`).
2. **Optional wait** (`wait_for_clear=True`). The call polls until no model is resident, or its own model is. There is **no time limit**: the resident model belongs to a phase that is still working, and a batch phase takes as long as its input needs. (A 120 s limit, added in `3bfde2d` and removed again, failed a 74-minute ingestion whose visual analysis took 174 s.)
3. **Swap.** Every other resident model is released. This makes the pool effectively single-slot: acquiring model B evicts model A.
4. **Deep flush** of CUDA memory.
5. **Admission control.** `required = on-disk size of model_path + learned backpack + 500 MB`, checked against free VRAM; if it doesn't fit, `OutOfVramError` is raised.
   - `required_vram_mb` / `<FEATURE>_VRAM_REQUIREMENT_MB` is **only used when `model_path` doesn't exist**.
   - The "backpack" is the peak activation memory beyond the weights. A telemetry thread samples it every 0.5 s and keeps the largest value per model path.
   - The backpack lives **in memory only** and starts at 0 in every new process. The first load of a model with large activations can therefore pass admission and still OOM during inference. Keep batch sizes configurable in `config.py`.
6. **`loader.load(manifest)`**, then the model is registered. A load exception flushes memory and is mapped to a `ModelManagerError`.

### `release_model(loaded_model, loader, flush=True)`
- If the model isn't registered (already released), this is a **no-op**, so a double release is safe.
- Otherwise it calls `loader.unload()`, deregisters the model and flushes.

### `release_all`
Evicts everything. `OnyxEngine` calls it (through `AsyncModelService.release_all`) after every ingestion and after any failed live call.

### "Reload"
There is no reload API.
- Reloading means calling `acquire_model` again after a release.
- Acquiring the same manifest while it is still resident is free (step 1), so a model can be reused across items and consecutive calls.
- A different `ModelType` pointing at the same file causes a full unload and reload.

### Preloads (background loads)
- `trigger_preload(manifest, loader)` submits `acquire_model(..., wait_for_clear=True)` to the **single** worker thread and returns a `Future`.
- A preload triggered while another model is resident **waits**; it never evicts a model that is in use. It loads once the current phase unloads.
- Several preloads queue up and load one after another, each waiting for the previous model to be released.
- `resolve_preload(future)` blocks until the load finishes and returns the `LoadedModel`, or raises the mapped `ModelManagerError`.
- `trigger_unload(...)` is **synchronous** and bypasses the worker. Never unload through the worker queue (`BackgroundModelWorker.unload_model`): it would sit behind a parked preload and deadlock.
- `discard_preload(future, loader)`: a queued preload is cancelled; a running or finished one has its model released as soon as it loads. This is what stops a halted pipeline from leaving a model loaded with nobody to release it.

## 3. Adapter (`app/features/<name>/data/<model>_adapter.py`)

The adapter implements a feature interface that extends `IModelLoader` (declared in the feature's `domain/interfaces.py`) and adds inference methods that take `model: LoadedModel`.

```python
class FooAdapter(IFooAdapter):  # IFooAdapter(IModelLoader, ABC) in domain/interfaces.py

    @staticmethod
    def _build_model(model_path: str) -> Any:
        from some_library import Model  # heavy import inside the loader, not at module import time
        device = "cuda" if torch.cuda.is_available() else "cpu"  # never hardcode "cuda:0"
        return Model(model_path, device=device, local_files_only=True)  # on-premise: never download

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Foo model missing: {manifest.model_path}")
            return LoadedModel(manifest=manifest, model_instance=self._build_model(manifest.model_path))
        except Exception as e:
            raise map_foo_error(e, context=f"Loading Foo model: {manifest.model_path}") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if getattr(model, "model_instance", None) is not None:
                if hasattr(model.model_instance, "close"):  # llama.cpp: free the C++ context explicitly
                    model.model_instance.close()
                model.model_instance = None  # drop EVERY reference (dict payloads: pop each entry first)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            raise map_foo_error(e, context="Unloading Foo model") from e

    def infer(self, model: LoadedModel, payload: str) -> str:
        if not getattr(model, "model_instance", None):
            raise FooInferenceError("Cannot run Foo: the model is not loaded.")
        ...
```

Rules:
- **Nothing loads in `__init__` or at import time.** Adapters are constructed once in bootstrap and must stay cheap. Only `load()` touches VRAM.
- **Never store the model on `self`.** It lives in `LoadedModel.model_instance`, which the orchestrator tracks.
- **`unload()` must release everything the model holds.**
  - llama.cpp: call `close()` before dropping the reference (`visual_cue_detection/data/qwen_detector_adapter.py`).
  - HF model plus processor in a dict payload: pop each entry (`visual_analysis/data/vlm_adapter.py`).
  - One leftover reference keeps the whole model in VRAM after it is "released".
- **sentence-transformers / torchvision imports** go inside `with block_torchcodec_import():` (see the reranker and both embedders). Otherwise a later `torchaudio.load()` in the same process can break.

## 4. Feature service API (`app/features/<name>/service/<name>_api.py`)

```python
class FooAPI:
    def __init__(self, orchestrator: ModelManagerOrchestrator, adapter: IFooAdapter):
        self._orchestrator = orchestrator
        self._adapter = adapter
        self._manifest = ModelManifest(
            model_type=ModelType.FOO,                          # this feature's OWN ModelType
            model_path=settings.FOO_MODEL_PATH,
            required_vram_mb=settings.FOO_VRAM_REQUIREMENT_MB
        )

    @property
    def manifest(self) -> ModelManifest:  # Phases preload with this
        return self._manifest

    def run(self, payload: str, release_after: bool = True) -> str:
        try:
            loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter)
            try:
                return self._adapter.infer(loaded_model, payload)
            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)
        except Exception as e:
            raise map_foo_error(e, context="FooAPI.run") from e
```

Rules:
- **Own `ModelType`.** Add one to `ModelType` in `app/core/model_manager/domain/enums.py`, even when you reuse another feature's weights file. Sharing a type would hand your code a model loaded with the other feature's settings.
- **Release in `finally`**, guarded by `release_after`. With `release_after=False` the model stays resident for the next call, so the owner (usually the Phase) must release it.
- **Batch APIs that take a pre-loaded model** follow `visual_cue_detection/service/detection_api.py`:
  - signature `process(..., model: Optional[LoadedModel] = None, release_after: bool = True)`;
  - set `manage_lifecycle = model is None`;
  - acquire only if `manage_lifecycle`;
  - release in `finally` only if `manage_lifecycle and loaded_model and release_after`.
- **Which acquire to use:**
  - **Ingestion / batch APIs** use the default `acquire_model(manifest, adapter)`, which evicts whatever is loaded. Batch pipelines own the GPU while they run.
  - **Live query APIs** (`RetrievalAPI`, `RerankAPI`, `RetrievalScrubberAPI`, `QAGenerationAPI`) use `wait_for_clear=True`, so a live call never evicts a model that is in use.
  - Consequence for `wait_for_clear=True` callers: if something else is still loaded and nobody releases it (e.g. an earlier call ran with `release_after=False` outside a Phase), the call **waits forever**. Anything that composes these APIs directly, such as an agent, must release each model before calling the next API.
- **Optional models** (weights may be absent) expose `is_available()` (the manifest path exists) and degrade instead of failing. `RerankAPI.rerank` returns chunks in RRF order without loading anything.
- **Let model errors through.** The feature's `utils/error_mapper.py` must return or re-raise `ModelManagerError` unchanged, never wrap it (see `rag_reranking`, `visual_analysis`, `qa_generation`). Phases and the engine rely on catching `ModelManagerError`.

## 5. Orchestration Phase (`app/orchestration/service/phases/<name>_phase.py`)

Copy `rag_reranking_phase.py`. The lifecycle-relevant skeleton:

```python
class FooPhase:
    def __init__(self, job_manager: JobManager, foo_api: FooAPI,
                 async_model_service: AsyncModelService, loader: IModelLoader):
        ...
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        # (optional model: `if not self._foo_api.is_available(): return`)
        self._preload_future = self._async_service.trigger_preload(
            manifest=self._foo_api.manifest, loader=self._loader
        )

    def discard_preload(self) -> None:
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._loader)

    def execute(self, items) -> PhaseBatchResult:
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None   # clear BEFORE resolving
            try:
                loaded_model = self._async_service.resolve_preload(future)
            except ModelManagerError as e:
                return PhaseBatchResult(phase=PipelinePhase.FOO, status=PhaseStatus.FAILED,
                                        errors=[f"Critical failure preloading Foo model: {e}"])

        try:
            for item in items:
                self._job_manager.run_job(
                    request, self._foo_api.run, payload=item.payload,
                    # keep the model across items only when this phase owns a resolved preload
                    release_after=loaded_model is None
                )
        finally:
            if loaded_model:
                self._async_service.trigger_unload(loaded_model=loaded_model, loader=self._loader, flush=True)
```

Rules:
- **The `loader` is the same adapter instance** the API was built with. In bootstrap that looks like `RagRerankingPhase(job_manager, rerank_api, async_service, reranker_adapter)`.
- **Clear `_preload_future` when resolving it** (the swap line above), so `discard_preload()` afterwards is a no-op and can't release a model the phase is using.
- **Per-item failures** become failed item results. Only a failed preload fails the whole phase.
- **Unload in `finally`** with `trigger_unload(..., flush=True)`, whether items succeeded or not.

## 6. Pipeline (`app/orchestration/service/pipelines/`)

- **Overlap loading with CPU work.** Call the *next* GPU phase's `preload_model()` just before a CPU-bound phase runs, so the weights load while the CPU works. `batch_media_pipeline.py` does this three times:
  - VAD preload → audio extraction;
  - VLM preload → frame extraction;
  - embedder preload → context unification.
- **Preloading ahead of a GPU phase is also safe.** A preload triggered while the current phase's model is loaded waits until that phase unloads, however long it runs, and never evicts it (e.g. `visual_hallucination_correction_phase.preload_model()` runs before `visual_analysis_phase.execute(...)`).
- **Release a halted branch's preloads before the next preload.** If a phase fails or has nothing to do, the preloads already triggered for the phases after it still load, and nothing releases them until `finally`. Any later preload would wait behind them for the rest of the run, so call `_discard_pending_preloads()` before it (`batch_media_pipeline.py` does this right before the embedder preload).
- **Live pipelines preload every phase up front** (`preload_models()`). The worker then loads them one at a time, as each earlier phase unloads.
- **Discard in `finally`.** Every GPU phase must be listed in the pipeline's `_discard_pending_preloads()`, which is called in `finally`. It covers every early return and exception, so no preload outlives the run. A new GPU phase that isn't in this list leaks its model whenever the pipeline stops before reaching it.

## 7. Engine, config and wiring

- **`app/core/config.py`:**
  - `FOO_MODEL_PATH` and `FOO_VRAM_REQUIREMENT_MB` (a conservative estimate; see §2 for when it is used);
  - batch-size settings for anything whose activation memory scales with input.
- **`app/engine/service/bootstrap.py`:**
  - use the one shared `orchestrator` / `async_service`, never construct another;
  - construct the adapter once and pass that instance to both the API and the Phase.
- **`OnyxEngine`:** a new top-level entry point must release all models when it fails. Follow `_release_models_after_failure()` (live calls) and `_cleanup_after_ingestion()` (batch).
- **Hardware:** there is no device selection yet (single shared VRAM pool, no `device_id`). Don't hardcode `cuda:0`, and don't invent a multi-GPU abstraction unless the task asks for it.

## 8. Anti-patterns

- Calling `loader.load()` / `loader.unload()` directly instead of going through the orchestrator.
- Keeping `LoadedModel` or `model_instance` on `self` between calls.
- Expecting two models to be loaded at once. Acquiring the second evicts the first.
- Reusing another feature's `ModelType` for shared weights.
- Acquiring without a `finally` release, or setting `release_after=False` with no owner that releases later.
- A Phase that doesn't clear `_preload_future`, lacks `discard_preload()`, or isn't registered in the pipeline's `_discard_pending_preloads()`.
- Unloading through the worker queue instead of `trigger_unload`.
- Wrapping `ModelManagerError` inside a feature exception.
- Calling a `wait_for_clear=True` live API while your own model is still loaded: it waits forever.
- Triggering a preload while a halted branch's preload is still loaded, without discarding it first: the new preload waits forever.
- Downloading at runtime: always use `local_files_only=True` and local paths.

## 9. Tests

**Unit tests** (mocked `orchestrator` / `adapter` / `async_service`, exact call arguments pinned). Mirror the reranker tests.

- **API** (`test_rerank_api.py`):
  - the manifest comes from settings;
  - `acquire_model` is called with exactly `(api.manifest, adapter, ...)`, using the right `wait_for_clear` convention;
  - `release_model(loaded, adapter, flush=True)` runs on success **and** when inference raises (use `side_effect`);
  - `release_after=False` → `release_model.assert_not_called()`;
  - weights missing or empty input → `acquire_model.assert_not_called()`.
- **Phase** (`test_rag_reranking_phase_unit.py`):
  - `preload_model()` calls `trigger_preload(manifest=..., loader=...)`, or doesn't call it for an unavailable optional model;
  - with a resolved preload: `run_job(..., release_after=False)` and `trigger_unload(loaded_model=..., loader=..., flush=True)`;
  - without a preload: `release_after=True` and `trigger_unload.assert_not_called()`;
  - `resolve_preload` raising `ModelManagerError` → phase status `FAILED`;
  - `discard_preload()` → `async_service.discard_preload(future, loader)`.
- **Pipeline:** when an early phase fails, every later phase's `discard_preload()` is still called.

**Integration tests** (`integration_tests/features/<name>/`):
- Skip with `pytest.skip(...)` when `settings.FOO_MODEL_PATH` doesn't exist.
- Build a real `ModelManagerOrchestrator(PyTorchVramMonitor())`, acquire through the API or orchestrator, and release in `finally`.
- Run GPU integration files in separate processes (one `pytest` per file) so models and the torchcodec guard state don't leak between tests.

## 10. Checklist for a new model-backed feature

- [ ] `ModelType.FOO` added; `FOO_MODEL_PATH` / `FOO_VRAM_REQUIREMENT_MB` (and batch sizes) in `config.py`
- [ ] Adapter implements `IModelLoader`; nothing loads in `__init__`; `unload()` drops every reference and empties the CUDA cache
- [ ] API builds its own manifest, exposes `manifest`, releases in `finally` behind `release_after`, and lets `ModelManagerError` through
- [ ] Phase: `preload_model` / `discard_preload` / resolve-and-clear / `release_after=loaded_model is None` / `trigger_unload` in `finally`
- [ ] Pipeline: preload placed before CPU work where possible; phase registered in `_discard_pending_preloads()`
- [ ] Bootstrap: shared orchestrator/async service; one adapter instance passed to both the API and the Phase
- [ ] Unit tests for API and Phase lifecycle; self-skipping integration test with a real orchestrator
