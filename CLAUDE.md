# CLAUDE.md

Guidance for Claude Code (and any engineer) working in this repository. Read this before touching `app/`.

## What Onyx is

Onyx is a fully **on-premise AI appliance** sold to law firms, hospitals, and other businesses that cannot let their data touch a third-party cloud. A client buys the appliance, points it at their local media/documents, and it:

1. Ingests that data through a local pipeline (transcription, diarization, visual analysis, correction).
2. Indexes it into a local RAG store (PostgreSQL + pgvector, hybrid dense + sparse search).
3. Lets people ask questions of a local LLM that answers **only** from that indexed data, with a citation back to the source chunk for every claim it makes.

The non-negotiable product requirement: **the model must never recall or state something that isn't grounded in the client's ingested data, and every answer must be traceable to a source.** A lawyer or doctor relying on a hallucinated answer is the failure mode this whole architecture exists to prevent. This is why the pipeline has multiple explicit correction/scrubbing stages (hallucination correction, retrieval hallucination scrubbing, visual hallucination correction) rather than trusting any single model pass.

No data leaves the premises — there is no cloud API call anywhere in the inference path. It's sold as appliance hardware, not a hosted SaaS product.

## Hardware reality — this drives the whole architecture

- **Dev machine (current):** 1× NVIDIA RTX 3060, **12GB VRAM**, AMD Ryzen 7 2700X, 32GB RAM @ 3200MHz.
- **Target client hardware:** 2× NVIDIA RTX 3060, **24GB VRAM total**, intended to be split so one GPU serves the **ingestion pipeline** and the other serves the **LLM** continuously for live chat.

Because no single GPU here can hold more than one or two of this pipeline's models at once, **nothing in this codebase may assume a model stays resident in VRAM.** Every GPU-bound component loads its model just-in-time, uses it, and releases it. This is not an optimization to add later — it is the baseline design constraint for every feature that touches the GPU. See "Model management rules" below.

The flip side: today the codebase also does not have any concept of *which* GPU a model runs on — there is no `device_id` or `CUDA_VISIBLE_DEVICES` anywhere in `app/core/model_manager/` or `app/core/config.py`. Everything assumes one shared VRAM pool on one device, which matches the dev box but not the target 2-GPU appliance. **Do not silently bake in single-GPU assumptions in new code** (e.g. don't hardcode `cuda:0`) — but also don't try to build out the full dual-GPU pool abstraction speculatively unless a task specifically asks for it. Flag it when it becomes relevant.

## Architecture: feature-first, hexagonal per feature

Every feature — under `app/core/`, `app/features/`, `app/legacy_features/`, and `app/tools/` — follows the **same internal folder convention**, ports-and-adapters style:

```
<feature>/
├── __init__.py
├── domain/          # entities.py, interfaces.py (ports/ABCs), enums.py — pure, no I/O, no framework imports
├── data/             # adapters implementing domain interfaces: model adapters, DB repositories
├── service/          # the feature's public API (e.g. retrieval_api.py) — the ONLY entrypoint other code should call
├── exceptions/        # base.py + specific typed exception modules (database.py, model.py, vram.py, network.py, ...)
└── utils/             # feature-local helpers, almost always including error_mapper.py
```

- `domain/interfaces.py` defines the contracts (e.g. `ITranscriptionAdapter`, `IRetrievalRepository`, `IModelLoader`) — these are what make features swappable and testable in isolation.
- `data/` implements those interfaces against real infrastructure (a specific model library, a specific Postgres table).
- `service/*_api.py` is what everything outside the feature is allowed to call. Never reach into another feature's `data/` directly.
- `exceptions/` + `utils/error_mapper.py`: every feature translates low-level/third-party exceptions (DB errors, CUDA OOM, file-not-found) into its own typed exception hierarchy at the boundary, so callers up the stack only ever see feature-level exceptions.
- Simple features with no persistence or model adapter (e.g. `hallucination_correction`, `scanner`) legitimately omit `data/`. Don't add an empty `data/` folder just for consistency — only add it if there's a real adapter to put there.

### Composition layers above individual features

- **`app/orchestration/service/phases/`** — one `Phase` class per pipeline stage. A Phase wraps a feature's `service` API with job tracking (`JobManager.run_job`) and, for GPU-bound stages, model preload/unload calls around the feature's work.
- **`app/orchestration/service/pipelines/`** — chains Phases into full workflows: `BatchMediaPipeline` / `BatchAudioPipeline` (ingestion), `LiveRetrievalPipeline` / `LiveVerifiedRetrievalPipeline` / `LiveQAPipeline` / `LiveReasoningPipeline` (query-time), and `DeepResearchReportPipeline` (a whole multi-page report per run).
- **`app/engine/`** — the composition root. `OnyxBootstrapper.build()` (`service/bootstrap.py`) is the DI container that wires every adapter/API/Phase/Pipeline together and constructs the single shared `ModelManagerOrchestrator`. `OnyxEngine` (`service/onyx_engine.py`) is the top-level facade: `start()/stop()`, `ingest_videos()/ingest_audios()`, `search_standard()/search_verified()/chat()/chat_reasoned()`, and `propose_research_plan()/execute_research_report()`. It holds an `EngineState` lock that makes batch ingestion, report runs and live search/chat **mutually exclusive** — ingestion takes `INGESTING` and a report takes `RESEARCHING`, and either rejects live calls with `EngineBusyError` until it's done. This exists because ingestion and inference currently compete for the same single GPU; splitting them across the two target GPUs is the natural way to lift this restriction, but that work hasn't been done yet.

### How to add a new feature

1. Define the domain interface(s) in `domain/interfaces.py` first — decide the contract before the implementation.
2. Implement the adapter(s) in `data/` against that interface (model wrapper, DB repository). If the feature loads a model, the adapter implements `IModelLoader`.
3. Add a typed exception module under `exceptions/` and wire translation logic into `utils/error_mapper.py`.
4. Expose a `service/<name>_api.py` — this is the only thing other code (a Phase, or another feature) should import.
5. If the feature loads a model: add its path + `_VRAM_REQUIREMENT_MB` to `app/core/config.py` (`Settings`), following the existing naming pattern (`<FEATURE>_MODEL_PATH` / `<FEATURE>_VRAM_REQUIREMENT_MB`).
6. Register the feature's adapters/APIs in `OnyxBootstrapper.build()`.
7. If it belongs in a pipeline: add an orchestration `Phase` (`app/orchestration/service/phases/`) and register it in the relevant `Pipeline` (`app/orchestration/service/pipelines/`).
8. Add a unit test directory mirroring the new feature under `tests/features/<name>/...` (see Testing below) — don't skip this even for a first pass; an empty/stubbed test file is better than a missing directory, since the test tree's shape is expected to mirror `app/` exactly.

Use the **`scaffold-onyx-feature`** skill (`.claude/skills/scaffold-onyx-feature/`) to generate steps 1–4 and the checklist for 5–8 automatically — it seeds files from `app/features/hallucination_correction` (no-adapter template) or `app/features/rag_retrieval` (model/DB-adapter template) depending on the feature's shape.

## Model management rules

All GPU/VRAM-consuming code must go through `app/core/model_manager/`:

- Every model adapter implements `IModelLoader` (`load(manifest)` / `unload()`).
- Acquire and release models only via `ModelManagerOrchestrator.acquire_model(manifest, loader)` / `release_model(loaded_model, loader)`. Never hold a raw model reference across a call boundary outside that lifecycle — if the orchestrator doesn't know a model is loaded, it can't budget for it or evict it.
- **Swap-on-acquire**: calling `acquire_model` evicts every other currently-resident model before loading the new one (this is effectively a single-slot pool today, matching the single-GPU dev box). Don't assume two models can be resident simultaneously unless you've confirmed the target has the VRAM budget and the orchestrator has been extended to support it.
- **Preload-ahead**: Phases that know they'll need a model soon can call `AsyncModelService.trigger_preload()` to start loading it on a background thread (`BackgroundModelWorker`, a 1-worker `ThreadPoolExecutor`) while other I/O-bound work continues, then `resolve_preload()` to block on it when actually needed. `trigger_unload()` is deliberately synchronous (bypasses the worker pool) so it can't deadlock behind a parked preload task.
- Admission control (`OutOfVramError`) budgets **on-disk weight size + learned "backpack" + 500 MB**. The backpack (peak activation/overhead memory beyond the weights, per model path) is learned by a telemetry thread and kept in memory only, so it starts at 0 on every process start. `<FEATURE>_VRAM_REQUIREMENT_MB` is only used when the weights path doesn't exist — still set it, conservatively.
- Always release what you acquire, in a `finally` block, so a failure mid-feature doesn't leak VRAM for the rest of the process.
- **`acquire_model` hands out a shared instance, not a refcounted handle** — `release_model` frees on the first call no matter how many callers hold it. That is deliberate (`release_after=False` makes acquires outnumber releases by design), so any release made *on behalf of* someone else must go through `release_preload`, which checks under the orchestrator's lock whether anyone has taken the model since. Freeing one that is in use aborts the process inside llama.cpp's `free_model` rather than raising.
- Before writing or changing code that loads a model, follow the **`onyx-model-lifecycle`** skill (`.claude/skills/onyx-model-lifecycle/SKILL.md`): adapter, feature API, Phase and pipeline patterns, preload/discard, and how to test them.

## Languages

An appliance serves **one** language, `APPLIANCE_LANGUAGE` (`en`, `sr`, `sl`). Every rule that depends on the spoken language lives in its pack, `data/languages/<code>.toml`, served by `app/core/language` (`get_language_pack()`); see `docs/core/language.md`. Nothing is translated — transcript, index and answers stay in the spoken language, because a translated index would put model-written text under every citation.

- **Never hard-code language again.** An English word list, an `[a-z]` class, a refusal sentence, a Whisper language code or a stemmer name in new code is a bug on a Serbian appliance. Add a field to the pack instead (`LanguagePack` validates it at startup), and give `en.toml` the value the code used to hard-code, so English doesn't move.
- **Prompts stay English.** Wrap a prompt whose output is stored or shown in `pack.localize_prompt()`; it's a no-op for English. Don't write per-language prompt copies.
- **A pack's `[models]` table overrides Settings.** Serbian and Slovenian swap Llama 3.2 3B and Qwen2.5-3B for Gemma 4 E4B. An explicit env value always wins.
- **Unit tests switch language with the `appliance_language` fixture** (`tests/conftest.py`), which changes it for every module at once; `language_pack(code)` loads a shipped pack.



- **Language / runtime:** Python, no web framework and no server anywhere — `app/main.py` is a CLI entrypoint
  (its own numbered menu, driving `OnyxEngine`), and `frontend/main.py` is a PySide6 (Qt6) desktop app that drives the same
  `OnyxEngine` **in-process** on a background thread. See `docs/frontend.md` and `frontend/README.md`.
- **Storage:** PostgreSQL + `pgvector`, a **single database for both relational and vector data** (jobs, transcripts, and the `vector_chunks` table with both an `embedding` column and an auto-generated `fts_document` tsvector column live side by side). Run via Docker: `docker-compose.yaml` defines `db_dev` (port 5432, `pgvector/pgvector:pg15`) and `db_test` (port 5433), both bootstrapped from `scripts/init.sql`. **There is no application Dockerfile** — only the databases are containerized; the app itself runs from a local venv.
- **ORM / DB access:** SQLAlchemy + `psycopg2-binary` (sync, not async).
- **Audio pipeline:** PyTorch, NVIDIA NeMo (VAD via MarbleNet, diarization via TitaNet), faster-whisper (domain-adaptation draft pass + main transcription).
- **Vision:** Qwen2-VL-7B via native Hugging Face Transformers + bitsandbytes (4-bit, pre-quantized to disk) — **deliberately not** llama.cpp/GGUF (see gotchas below).
- **Local LLM inference:** `llama-cpp-python` running GGUF checkpoints (Llama 3.2 3B for lightweight NLP/classification tasks — Gemma 4 E4B on Serbian/Slovenian appliances, Qwen 2.5 for correction and retrieval scrubbing, Qwen3.5-9B for QA generation and the Deep Research reasoner), `n_gpu_layers=-1`. Qwen3.5 is a hybrid Gated DeltaNet / attention model: only 8 of its 32 layers keep a KV cache, which is why it runs at a 128K context (`QA_GENERATION_N_CTX`, q8_0 KV cache) on a 12 GB card. Chat calls go through `app/core/model_manager/utils/llama_cpp_chat.py`, which passes `enable_thinking` to the GGUF chat template (`Llama.create_chat_completion` drops template variables).
- **Embeddings:** `sentence-transformers` (BGE-M3), fp16 on CUDA.
- **Reranking:** `sentence-transformers` `CrossEncoder` (bge-reranker-v2-m3, optional — without weights at `RAG_RERANKER_MODEL_PATH` search keeps RRF order).
- **Sentence segmentation:** `wtpsplit` (SaT).
- **Config:** `pydantic-settings`, environment selected by `ENV` (`development`/`test`/`production`) loading `.env` / `.env.test` / `.env.prod` respectively.

## Testing: unit vs integration

- **`tests/` (unit)** — mirrors `app/`'s full internal layering per feature (`data/`, `service/`, `utils/`, `domain/` subfolders match exactly). Fully mocked with `unittest.mock` (`patch`, `MagicMock`) — no real DB, GPU, filesystem, or network. One `TestXxx` class per module under test, methods named `test_<behavior>`, assertions pin down exact call arguments (not just "was called"). Error paths are tested with `pytest.raises(<TypedException>)`. `utils/error_mapper.py` is almost always tested with `@pytest.mark.parametrize` tables.
  Run: `pytest tests`
- **`integration_tests/` (integration)** — flat, one or two files per feature (no internal layering split), plus a root `conftest.py`, plus a `resources/` folder of real fixture media (audio/video/images/JSON transcripts). Uses a **real** Dockerized Postgres (`db_test`, port 5433): a session-scoped `autouse` fixture waits for Docker to be reachable, drops/recreates the schema, re-enables the `vector` extension, and runs `Base.metadata.create_all()`; a function-scoped `db_manager` fixture wraps each test in a real transaction + SAVEPOINT that's rolled back afterward for isolation. Real models/GPU are used where available; tests that need hardware or weights not present on the current machine call `pytest.skip(...)` with a descriptive reason rather than failing. Marked with `@pytest.mark.integration` (registered in `pytest.ini`).
  Run: `docker compose up -d db_test`, then `pytest integration_tests -m integration`
- **`frontend/tests/` (desktop unit)** — mirrors `frontend/onyx_desktop/`, fully mocked, run headless under
  `QT_QPA_PLATFORM=offscreen`. Deliberately has **no `__init__.py` files**: a package named `tests` there would
  shadow the root `tests` package and break collection of the whole existing suite, so filenames carry a `_unit`
  suffix to stay globally unique.
  Run: `QT_QPA_PLATFORM=offscreen pytest frontend/tests`
- **Both:** bare `pytest` runs everything (`pytest.ini`'s `testpaths = tests integration_tests frontend/tests`).
- No CI is configured yet — these commands must be run manually.
- When adding a feature: write unit tests immediately (mirrored directory, mocked). Add an integration test if the feature touches the DB or loads a real model, and make it self-skip gracefully when the resource isn't available rather than failing hard.

## Known gotchas

- **`sys.modules["torchcodec"] = None` hack** in `app/core/model_manager/utils/torchcodec_guard.py` (`block_torchcodec_import()`, used by both embedders and the reranker): scoped to work around a torchvision/torchcodec crash when loading the sentence-transformers embedder, restored in `finally`. `integration_tests/core/model_manager/test_torchcodec_environment_integration.py` proves this can still leak across features within one long-lived process (load the RAG embedder, then later call `torchaudio.load()` for VAD in the same process → `ImportError: TorchCodec is required...`). This is directly relevant to anything long-running that mixes retrieval and ingestion in one process — the RAG agent being built next is exactly that shape. Don't assume this hack is fully process-safe.
- **Qwen-VL must stay on native HF Transformers, not llama.cpp/GGUF** — llama.cpp's `mtmd` vision encoder has a confirmed, unresolved upstream SIGSEGV in the Qwen-VL vision graph allocator. See `app/features/visual_analysis/data/vlm_adapter.py` and `integration_tests/features/visual_analysis/test_vlm_multimodal_diagnostics.py` for the documented regression evidence. Don't "simplify" the vision adapter onto llama.cpp for consistency with the other LLM adapters.
- **Every chat turn re-reads the whole conversation.** The embedder, reranker and scrubber are swapped in between turns, so the answer model is reloaded and prefills the full history each time (tens of seconds at 50K+ tokens on a 3060). Messages are built append-only so a future KV-state cache (`save_state`/`load_state`, or a dedicated LLM GPU) can reuse the prefix; hybrid models like Qwen3.5 can only reuse state when the new prompt strictly extends the old one.
- **Report mode's model-swap schedule is load-bearing, and a test enforces it.** `ReportAuthoringPhase` is deliberately ONE `execute_report()` covering outline optimisation, every section write and the summary, because each `execute` resolves a preload and unloads in its own `finally` — splitting it in two would mean two loads of the 5.7 GB GGUF instead of one. Likewise `RetrievalAPI.search_batch` acquires the embedder once for a whole fan-out. A full run costs ~6 model loads (2 heavy); a naive per-section arrangement of the same work costs ~40 (16 heavy). `integration_tests/orchestration/test_deep_research_report_pipeline_integration.py` counts the loads, so a "simplification" here fails loudly rather than silently costing minutes.
- **An unresolved preload can wedge the process, silently.** `Phase.preload_model()` stores a future in `self._preload_future`. If anything returns or raises between firing a preload and resolving it, the next call overwrites that future and the old one becomes **unreachable** — the background worker still loads the model, `discard_preload` can never be called on it, nothing releases it, and the next `acquire_model(..., wait_for_clear=True)` parks forever with no exception and no exit but a restart. `discard_preload` cannot rescue it: `future.cancel()` fails once the task is running, and its done-callback only fires after the load completes, which needs the VRAM it is waiting on. Straight-line pipelines resolve every preload exactly once so this never fired; **any loop must self-guard** (the four report-path phases call `discard_preload()` at the top of `preload_model()`) and discard per iteration, not only in the run's `finally`.
- **`QComboBox.currentData()` does not give you back your enum.** userData is stored as a QVariant, and a `str`-derived `Enum` comes back out as a **plain `str`**. Every gateway branch compares with `is`, so this silently made Reasoned Answer run single-pass chat, Verified search run standard search, and Audio ingest run the video pipeline — all three shipped that way. Each view now rebuilds the enum from `currentData()` at the boundary; keep doing that rather than loosening the `is` comparisons.
- **Switching `APPLIANCE_LANGUAGE` on an existing database fails `OnyxEngine.start()`** (`IndexLanguageMismatchError`). `vector_chunks.fts_document` is generated with the old language's stemmer, and there are no migrations: reset the database and re-ingest. Likewise, a file Whisper confidently hears in another language fails transcription (`LanguageMismatchError`) rather than being forced into the appliance's language, which would make Whisper translate or invent text.
- **Ingestion and live inference are currently mutually exclusive** — `OnyxEngine`'s state lock blocks `search_*`/`chat` while a batch ingestion job holds `INGESTING`. This is a single-GPU-era simplification, not a permanent design decision.

## Where things stand / what's next

Ingestion (media → transcript → RAG index, including a visual-analysis branch) and retrieval (hybrid RAG search → cross-encoder reranking → verified/hallucination-scrubbed search → cited QA generation via a local Qwen3.5-9B model, which reads only verbatim speaker-labelled transcript text) are both built and wired into `OnyxEngine`. See `CHANGELOG.md` for the feature-by-feature history.

**Four query-time modes, all sharing one citation scheme:**

1. **`chat(query, session_id)`** — single-pass retrieve → scrub → generate, multi-turn. `session_turns` stores each turn with the evidence it showed first, and `[Doc N]` numbers are conversation-wide (`first_doc_number` per turn), so a follow-up can cite an earlier turn's excerpt.
2. **`chat_reasoned(session_id, query)`** — the Reasoner loop (`LiveReasoningPipeline`): multiple retrieval passes judged by a Qwen3.5-9B reasoner, then draft + self-critique. Still ends in one chat answer. This was called "Deep Research" until the report mode took the name; `AGENT_DESIGN.md` describes it under the old one.
3. **`propose_research_plan()` → `execute_research_report()`** — Deep Research Report mode (`app/features/deep_research_report/`, `DeepResearchReportPipeline`). A topic in, a multi-page cited Markdown report out, via a WebWeaver-style planner/writer split: sweep the corpus into a numbered evidence bank, assign it to an outline, then write **section by section**, each section seeing only its own evidence. Two calls because the run takes tens of minutes and stops for outline approval first. Holds the GPU exclusively (`EngineState.RESEARCHING`).
4. **`execute_iterative_research_report()`** — Exhaustive mode. Same pipeline, but gathering repeats while sections come back unsupported, up to `REPORT_MAX_GATHER_ROUNDS`. **Code decides *when* to loop** (a section got no evidence; a round added no new chunks); the model only decides *what to ask next*, re-wording a failed query with vocabulary now visible in the evidence bank. Without it, "not covered" conflates "the corpus lacks this" with "the sub-question was worded badly".

**When working on report mode, the things to keep in mind:**

- **`[Doc N]` is assigned once, report-wide**, in reading order, before any section is written — that is what makes a number mean the same thing across sections, the summary, the references table and a resumed run. Per-section citations are then validated against *that section's* allowed numbers.
- **Absence is a finding.** A section the corpus cannot support becomes `SKIPPED_NO_EVIDENCE` and is listed under "Not covered by the available material"; a report states `Coverage: swept N of M indexed sources` rather than implying it read everything. Never let it ship uncited prose instead.
- **One model acquisition per round, and the last round also writes.** `ReportAuthoringPhase.execute_report` takes `stop_on_gaps: bool` — a flag, deliberately not a callback. A callback invoked while the phase holds the model, whose purpose is to decide "go gather more", would hand the next reader the exact footgun the design avoids: re-gathering from inside the residency parks forever on `wait_for_clear`. A one-round exhaustive run therefore costs exactly what a deep run costs.
- **A gathering round must have an unasked query, or it is not entered.** The run keeps a `QueryAttempt` per search (text, round, chunks it added that weren't already banked). The model sees it in the outline prompt so it can re-word rather than paraphrase, and `ReportAuthoringAPI` filters follow-ups against it — the fallback follow-up is derived from a section's title and brief, which don't change between rounds, so without the filter an uncovered section re-issues the identical query until the budget runs out, at three model loads a round. No follow-ups is how the API says "stop". The history is passed only to a pass that can act on it, so deep mode's prompt is unchanged.
- **`research_report_evidence.chunk_id` is intentionally not a foreign key** — re-ingesting a file cascades its chunks away, and a signed-off report must still render its citations from the denormalised provenance.

**Where the desktop app is.** Four views — Chat, Search, Ingest, Report. The Report view (`frontend/onyx_desktop/ui/views/report_view.py`) does the whole loop: topic + mode, an editable outline approval step, a determinate progress bar, a working Stop, the rendered report with **genuinely clickable `[Doc N]` citations**, and Save as Markdown. Two things about it are worth knowing before changing it:

- **Its citations are the app's only real click-to-verify on prose.** The chat view's chips are `<span>`s and do nothing when clicked — you verify a chat claim through the evidence card beneath it. The report uses `QTextBrowser` + `setOpenLinks(False)` + `anchorClicked` against the `onyx://doc/N` scheme `markdown_renderer` emits in link mode.
- **Cancel bypasses `EngineBridge`'s worker queue**, the one deliberate exception to "everything through the worker". The worker is busy running the report being stopped, so a queued cancel would arrive exactly when it is useless. Safe only because `cancel_research_report` just sets a `threading.Event`.

**What's next: PDF/DOCX export.** Both should be generated *from the Markdown*, never from the model. The dependency review matters on an air-gapped appliance: PDF via `QTextDocument` + `QPdfWriter` needs nothing new (QtGui is already in `PySide6-Essentials`), and DOCX via `python-docx` is pure Python with no native deps. Do not reach for `reportlab`, `weasyprint` or `pandoc` — native deps or a binary, on a box that ships without a package manager.
