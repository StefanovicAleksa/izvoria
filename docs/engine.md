# engine

`app/engine/`

The composition root and top-level facade of the whole application. Two pieces: `OnyxBootstrapper` (wires everything together) and `OnyxEngine` (the object every caller — CLI, and eventually the RAG agent — actually talks to).

## `OnyxBootstrapper.build()`

`app/engine/service/bootstrap.py`. The single dependency-injection root. In order, it constructs:

1. Core infra: `JobRepository`/`JobManager`, `PyTorchVramMonitor` → `ModelManagerOrchestrator` → `BackgroundModelWorker` → `AsyncModelService`.
2. Every feature's repository + adapter + `*API` object: storage, audio extraction, VAD, domain adaptation, transcription, hallucination correction, diarization, sentence reconstruction, domain phonetic correction, contextual retrieval, all five visual features, context unification, clip extraction, RAG storage, RAG retrieval, RAG reranking, retrieval scrubber, QA generation.
3. Every orchestration `Phase`, wrapping those APIs with `job_manager`/`async_service`.
4. The five `Pipeline`s (`BatchMediaPipeline`, `BatchAudioPipeline`, `LiveRetrievalPipeline`, `LiveVerifiedRetrievalPipeline`, `LiveQAPipeline`).
5. Finally assembles `OnyxEngine(...)` and returns it, typed as `IOnyxEngine`.

Notably, `HallucinationScrubberPhase` and `IntelligenceBridgePhase` (see `legacy_features.md`) are **not** imported or constructed here — confirming they are dead code at the wiring level, not just at the feature level.

## `OnyxEngine`

`app/engine/service/onyx_engine.py`, implements `IOnyxEngine`. The facade with an `EngineState` state machine (`OFFLINE`/`BOOTING`/`ONLINE`/`INGESTING`/`ERROR`) guarded by an `RLock`.

Public API surface:

- `start()` — migrates the database to the newest schema (`upgrade_to_head`, see `docs/core/database.md`), forces a GC/CUDA cache clear, transitions to `ONLINE`.
- `stop()` — GC, `async_service.shutdown()` (waits out any preload still loading), `async_service.release_all()` (frees every resident model, so none outlives the engine), `db_manager.dispose()`, transitions `OFFLINE`.
- `get_state() -> EngineState`.
- `active_language` — `settings.APPLIANCE_LANGUAGE`: the language new batches are indexed in and answers are written in. `require_idle()` raises `EngineBusyError` while `BOOTING`, `INGESTING` or `RESEARCHING`, the only times it may not change.
- **Switching language** is `OnyxBootstrapper.switch_language(engine, code, progress_listener)`, not an engine method. It runs `require_idle()`, then `stop()`, then `settings.use_language(code)`, then `build()` and `start()`, and remembers the language. It returns the new engine; a failed rebuild returns settings to the previous language. The engine is rebuilt because `build()` bakes model paths into manifests, and a language's pack may swap the small models. See `docs/core/language.md`.
- `ingest_videos(target_path) -> IngestionReport` / `ingest_audios(target_path) -> IngestionReport` — locks the engine into `INGESTING` (raises `EngineBusyError`/`EngineOfflineError` otherwise), delegates to `BatchMediaPipeline.process_directory()` / `BatchAudioPipeline.process_directory()`, aggregates per-phase results into scanned/success/failed/vectors/clips metrics plus a per-phase error dict, evicts every resident model (`AsyncModelService.release_all`) and always unlocks in `finally`.
- `search_standard(query) -> SearchResult` — delegates to `LiveRetrievalPipeline`, then hydrates `physical_path` on each result chunk via `StorageAPI`.
- `search_verified(query) -> SearchResult` — delegates to `LiveVerifiedRetrievalPipeline` (adds SLM hallucination scrubbing on top of standard search).
- `chat(query) -> QAResponse` — delegates to `LiveQAPipeline`, then hydrates each citation's `physical_path` the same way as search results (extracted clip via `extracted_file_id`, otherwise the source file).
- If any live call fails, the engine calls `release_all()` before re-raising, so the next request never waits on a model the failed call left resident.
- **History, read-side**: `list_conversations` / `get_conversation` / `delete_conversation`, `list_searches` / `get_search` / `delete_search`, next to `get_report` / `list_reports` / `get_chunks`. They only read the database, so they check only that the engine has started (`_require_started`), not `_enforce_online_state`. They work mid-ingestion and mid-report. Searches are recorded by `search_standard` / `search_verified` after they succeed, and a recording failure never fails the search. See `docs/features/search_history.md`.
- `plan_cost(plan) -> (expected_job_count, estimated_minutes)` sizes a research plan. `propose_research_plan` uses it, and so does the desktop app when it reopens an unapproved outline from history.
- There is no CLI on the engine itself. `app/main.py` owns the terminal menu and drives these methods; the desktop app drives the same ones through `frontend/onyx_desktop/data/engine_gateway.py`. (`OnyxEngine.interactive_cli()` was a second, stale copy of that menu — it had never been called and was missing Reasoned Answer and Report mode, and has been removed.)

## Gotchas / design notes

- **Ingestion and live inference are mutually exclusive.** `_enforce_online_state()` blocks `search_*`/`chat` calls while the engine is `INGESTING`, and blocks ingestion while a search/chat lock would conflict — both compete for the same shared GPU today. This is the main constraint an agent or a multi-GPU deployment would need to lift.
- `search_standard`/`search_verified`/`chat` use `object.__setattr__` to mutate frozen Pydantic result entities when hydrating file paths after the fact — a deliberate, explicit workaround for the immutable-entity convention used throughout the domain layer, not an oversight.
- `OnyxEngine` is the natural place a RAG agent would sit on top of — it's the existing stable single-pass façade (`search_standard`/`search_verified`/`chat`) that any richer multi-turn/tool-using agent would either call directly or partially reimplement for finer control.
