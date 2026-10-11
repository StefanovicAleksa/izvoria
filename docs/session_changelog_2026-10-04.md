# Session changelog: multi-user web version and local dev mode (2026-10-04)

This covers every change made in one long Claude Code session, written so you can review work you didn't watch being done. **Nothing is committed.** Everything is in the working tree on `development`.

What the session did:

1. Evaluated your plan for a multi-user, multi-firm hosted web version, asked clarifying questions, and wrote an implementation plan, which you approved. It is saved at `~/.claude/plans/what-i-want-to-deep-lampson.md`.
2. Implemented that plan:
   - changes inside `app/`, all behind settings that default to today's desktop behaviour;
   - a new `web_deploy/` folder (gateway, box API, ingestion worker, React frontend, provisioning, tests).
3. Built `llama-server` locally and ran part of the plan's "M0 spike" on your RTX 3060. It found two real bugs, now fixed.
4. Added, at your second request, a **local dev mode**: the same web version on the 12 GB dev GPU, served on your LAN, with ingestion taking the whole GPU while it runs.

**Size:**
- 53 existing files modified (+1,487 / −277 lines) across `app/`, `tests/`, `integration_tests/`, `CLAUDE.md`, `CHANGELOG.md`, `pytest.ini` and `.gitignore`.
- 9 new files in `app/`, `tests/` and `docs/`.
- 98 files in `web_deploy/`, excluding `node_modules`, `.venv` and the built app.

**Not touched:** your uncommitted work from before the session (`deploy/**`, the source-offer files in `frontend/izvoria_desktop/**` and `frontend/tests/**`, `docs/deploy.md`, `docs/frontend.md`). `web_deploy/` *imports* four `deploy/` modules but doesn't modify them (see 3.10).

`CLAUDE.md` already had uncommitted edits of yours before the session. My additions are the four paragraphs listed in 6.1, so its diff mixes both.

---

## Contents

1. [Decisions you made](#1-decisions-you-made)
2. [Changes inside `app/`](#2-changes-inside-app): settings, chat backend, orchestrator, engine, multi-firm, per-user history, per-question language
3. [The new `web_deploy/` folder](#3-the-new-web_deploy-folder)
4. [Local dev mode](#4-local-dev-mode-second-request)
5. [Bugs found by running it for real](#5-bugs-found-by-running-it-for-real)
6. [Docs, memory and config](#6-docs-memory-and-config)
7. [Side effects on your machine](#7-side-effects-on-your-machine)
8. [Tests: what exists and what was run](#8-tests-what-exists-and-what-was-run)
9. [Measurements](#9-measurements)
10. [Where the code differs from the approved plan](#10-where-the-code-differs-from-the-approved-plan)
11. [Known gaps and risks: what to review first](#11-known-gaps-and-risks-what-to-review-first)
12. [How to undo it](#12-how-to-undo-it)

---

## 1. Decisions you made

| Question | Your answer |
|---|---|
| What do users use? | A new web frontend. noVNC can't serve more than one user. |
| How parallel? | Truly parallel: Qwen3.5-9B under `llama-server` with parallel slots. |
| GPU setup (hosted) | One 24 GB card, split about 13 GB live and 10–11 GB ingestion. |
| Answer language | Picked per question by the user. |
| Where it lives | A new folder (`web_deploy/`), hosted on Hetzner plus a rented Vast GPU. `deploy/` (noVNC) stays. |
| Frontend stack | React + Vite + TypeScript. |
| Accounts | The owner creates firms and users. No self sign-up. |
| Box uptime | On-demand start and idle stop, kept. |
| Second request | A local version for personal testing on the 12 GB dev GPU, over the LAN, where ingestion pushes the chat models out. |

---

## 2. Changes inside `app/`

**Principle:** every new setting defaults to the old behaviour. With no new environment variables the desktop app and CLI behave as before. The full appliance unit suite (2,217 tests) and the report load-count integration test pass unchanged.

### 2.1 New settings (`app/core/config.py`)

| Setting | Default (appliance) | Hosted / local value | Meaning |
|---|---|---|---|
| `LLM_BACKEND` | `llama_cpp` | `llama_server` | Where the four Qwen chat roles run |
| `LLM_SERVER_URL` | `http://127.0.0.1:8081` | same | The llama-server address |
| `LLM_SERVER_KV_TOKENS` | `0` (read from server) | 196608 / 131072 | KV budget for admission control |
| `LLM_SERVER_TIMEOUT_S` | 1800 | same | Per-request HTTP timeout |
| `MODEL_PINNED_TYPES` | `[]` | embedder, reranker | Loaded once, never freed |
| `MODEL_EXTERNAL_TYPES` | `[]` | QA, reasoner, report, scrubber | Served by llama-server |
| `ENGINE_CONCURRENT_LIVE` | `false` | `true` | Live calls run concurrently; the orchestrator becomes strict |
| `REPORT_MAX_CONCURRENT` | `1` | `2` | Reports allowed at once (concurrent mode only) |
| `RETRIEVAL_SCRUBBER_CONCURRENCY` | `1` | `3` | Chunks judged in parallel |
| `VRAM_HEADROOM_MB` | `0` | 1500 (hosted ingestion child) | VRAM the orchestrator treats as taken |

### 2.2 A llama-server backend for the Qwen chat models

**New files:**
- `app/core/model_manager/data/llama_server_chat_model.py`:
  - `LlamaServerChatModel` is an OpenAI-compatible HTTP client. It uses stdlib `urllib`, so no new dependency.
  - It implements the subset of `llama_cpp.Llama` the adapters use: `tokenize`, `n_ctx`, `chat`, `close`.
  - It sends llama-cpp-python's sampling defaults explicitly, because the server's defaults differ.
  - It passes `enable_thinking` through `chat_template_kwargs`.
  - `n_ctx()` returns the smaller of the role's window and the server's.
  - `close()` does nothing; the server owns the weights.
  - `KvTokenBudget` is a first-come-first-served admission semaphore counted in tokens, shared per server URL. A request reserves its estimated prompt tokens plus `max_tokens`.
  - **`JSON_GRAMMAR`** (added after the spike, see 5.1): `response_format: json_object` / `json_schema` is translated into llama.cpp's own `json.gbnf` grammar.
- `app/core/model_manager/exceptions/server.py`:
  - `LlamaServerError` is raised when the server is unreachable or answers with an error.
  - `ResidencyError` is raised by strict mode (2.3).
  - Both are `ModelManagerError`s, so callers handle them as before. They are exported from `exceptions/__init__.py`.
- `app/core/model_manager/domain/interfaces.py`: new port `IChatModel`.

**Changed:**
- `app/core/model_manager/utils/llama_cpp_chat.py`:
  - `build_llama()` returns a `LlamaServerChatModel` when `LLM_BACKEND=llama_server`, and raises on an unknown backend.
  - `chat_completion()` calls `llm.chat(...)` when given an `IChatModel`.
  - With the default backend, both behave exactly as before.
- `app/features/retrieval_hallucination_scrubber/data/slm_adapter.py`. **This is a behaviour-adjacent change on the appliance too, so review it:**
  - It now builds its model through `build_llama(path, 4096, "f16")` and calls `chat_completion(..., enable_thinking=False, ...)`, instead of constructing `Llama(...)` and calling `create_chat_completion` itself.
  - On the appliance this is equivalent: same n_ctx, f16 KV (llama.cpp's default), flash attention, same sampling arguments, same `response_format`.
  - **New:** the reply passes through `strip_reasoning` before JSON parsing, so the scrubber can be a Qwen3.5 model.
  - **Difference:** `chat_completion` sets `repeat_penalty=1.0`, which `create_chat_completion` also defaulted to.
- The QA, reasoner and report adapters were not edited. They already went through `build_llama` / `chat_completion`, and the server client duck-types the `tokenize` / `n_ctx` calls they make.

### 2.3 Orchestrator: pinned, external and strict residency (`app/core/model_manager/service/orchestrator.py`)

**New constructor arguments:** `pinned_types`, `external_types`, `strict_resident`. All empty or false by default, so nothing changes for the appliance.

- **Pinned:**
  - `warm_up()` loads it once.
  - It is excluded from `_held_ids()` (so a waiting acquire never waits for it), from batch eviction and from `release_all`.
  - `release_model` and `release_preload` do nothing for it.
  - Only the new `shutdown_all()` frees it.
- **External:**
  - `_acquire_external()` creates one client per (type, path) with no VRAM budgeting and no flush.
  - It is never unloaded, except by `shutdown_all()`.
- **Strict** (on when `ENGINE_CONCURRENT_LIVE`): acquiring any other type raises `ResidencyError`.
- **Why this instead of a refcount:** `release_model` is deliberately not a refcount (CLAUDE.md). Concurrent requests sharing a model that can be freed is the documented llama.cpp abort. A model that can't be freed is safe to share.
- **`release_all`** now skips permanent models. It only flushes the CUDA cache if it actually unloaded something, or if there are no permanent types.
- **`warm_up` admission (fix from the real run, see 5.2):** a pinned model is admitted against its declared `required_vram_mb`, not its on-disk size. The on-disk estimate refused BGE-M3 (4.3 GB on disk, 1.1 GB loaded) next to llama-server.
- **`resident_types()`** is new, for health and diagnostics.

**New file:** `app/core/model_manager/data/headroom_vram_monitor.py`. `HeadroomVramMonitor` reports free VRAM minus a headroom, so the hosted ingestion child leaves the live models' share alone.

### 2.4 Request-safe shared objects

- **`app/orchestration/utils/preload_slot.py` (new):**
  - `PreloadSlot` is a per-thread data descriptor.
  - Seven live and report phases declare `_preload_future = PreloadSlot()`: rag_retrieval, rag_reranking, retrieval_scrubber, qa_generation, reasoner, research_scoping, report_authoring.
  - Concurrent requests share the Phase objects, so each needs its own preload slot. On the single-threaded desktop it behaves exactly like the old attribute.
  - **Assumption:** a preload is fired and resolved on the same thread. True for every pipeline today.
- **Embedder and reranker:** each adapter takes a `threading.Lock` around `encode` / `predict`. The HF fast tokenizer is not thread-safe ("Already borrowed"). The lock is uncontended on the desktop. Files: `rag_retrieval/data/embedding_adapter.py`, `rag_reranking/data/cross_encoder_adapter.py`.
- **Checked:** no live or report pipeline, phase, feature API or `JobManager` stores per-run state on itself, beyond the preload future above.

### 2.5 Engine concurrency (`app/engine/service/izvoria_engine.py`)

**New constructor arguments:** `database`, `owns_model_service` (default `True`), `concurrent_live` (default from settings), `report_max_concurrent`.

- **Every live call is counted.** `search_standard`, `search_verified`, `chat`, `chat_reasoned` and `propose_research_plan` now run inside `_live_call()`, which increments `_inflight`.
  - `chat`, `chat_reasoned` and `propose_research_plan` were split into a public wrapper and a private `_chat` / `_chat_reasoned` / `_propose_research_plan`.
  - **Visible on the desktop too:** `require_idle()` (used by language switching) now also refuses while a live call or report is in flight. The desktop runs one call at a time, so this can't trigger in practice.
- `busy_snapshot()` is new: `{"inflight", "reports_running"}`.
- **With `concurrent_live`:**
  - Live calls are checked only for OFFLINE.
  - Reports skip the RESEARCHING state lock and are capped at `REPORT_MAX_CONCURRENT`; the same report can't run twice.
  - Report cleanup does not `release_all`, and a failed live call releases nothing.
  - Ingestion is refused (`EngineBusyError`, "runs in the ingestion worker").
  - `_force_aggressive_garbage_collection` skips `torch.cuda.empty_cache()`, which would hand the pinned models' reserved activation memory to another process. It became an instance method.
- **`start()` / `stop()`:**
  - Migrations run on the engine's own database (`self._db`, falling back to the global `db_manager`).
  - `stop()` disposes that database, and only shuts down and frees models when `owns_model_service`.
- **`active_language`** now reads `active_language()` (2.8).
- **Owner parameters:** every chat, search, report and history method gained an optional `owner_id=None` (2.7).

### 2.6 Multiple firms in one process

- **`app/core/database/data/engine.py`, `service/manager.py`:** `SQLAlchemyEngine` and `DatabaseManager` take an optional `url`, `pool_size` and `max_overflow`. The `db_manager` singleton is unchanged.
- **`rag_storage/data/postgres_rag_repo.py`, `rag_retrieval/data/postgres_search_repo.py`:** these two repos imported the global `db_manager`. They now take an optional `database` and fall back to the global, looked up at call time so existing test patches still work.
- **`storage/data/file_manager.py`, `storage/service/storage_api.py`:** `FileManager(base_path=None)` and `StorageAPI(repository, file_manager=None)`, so each firm has its own vault. `FileScanner` / `scanner/utils/filters.py` were *not* changed: ingestion runs in a child process whose `ONYX_STORAGE_PATH` is the firm's vault.
- **`app/engine/service/bootstrap.py`:**
  - New `ModelKit` dataclass: orchestrator, worker, async service and the seven live-model adapters, plus `warm_up()` and `shutdown()`.
  - New `build_model_kit()`.
  - `build()` gained keyword-only `database=`, `vault_dir=`, `reports_dir=` and `kit=`. With none of them it builds exactly what it built before.
  - Every `db_manager` reference inside `build()` became the passed-in `db`.

### 2.7 Per-user history (`owner_id`) and migration 0005

- **New migration:** `app/core/database/data/migrations/versions/0005_history_owner.py`. It adds a nullable, indexed `owner_id VARCHAR(64)` to `session_turns`, `search_history` and `research_reports`.
  - It was autogenerated against a scratch database (`onyx_alembic_scratch`, since dropped) per `docs/core/database.md`.
  - `alembic check` reported no drift.
  - **It runs on every existing appliance database at its next start.** It only adds nullable columns and indexes.
- **Models, interfaces, repositories and APIs** of `session_memory`, `search_history` and `deep_research_report` take `owner_id: Optional[str] = None`:
  - `None` (the appliance) means unfiltered, exactly as before.
  - A value filters reads and deletes. Another user's conversation id reads as empty, and another user's report or search reads as `None`.
  - Reports get their owner through a new `set_owner()`, called right after `propose_plan`, rather than by threading a parameter through the pipeline.

### 2.8 Answer language per question (`app/core/language/service/language_api.py`)

- **New functions:**
  - `active_language()`;
  - `language_scope(code)`, a context manager over a `ContextVar`;
  - `submit_in_context(executor, fn, ...)`, which carries the override into thread pools.
- `get_language_pack()` now follows `active_language()`.
- `settings.APPLIANCE_LANGUAGE` reads on the live path were replaced in `report_api.py` (the report's stored language) and `izvoria_engine.active_language`. `storage_api.py` and `ingest_phase.py` still read settings, which is intentional: ingestion runs one job per child process.
- The appliance never sets an override, so nothing changes there.

### 2.9 Parallel scrubbing (`retrieval_hallucination_scrubber/service/scrubber_api.py`)

- The per-chunk logic moved into `_judge()` and `_judge_untrusted()`.
- With `RETRIEVAL_SCRUBBER_CONCURRENCY > 1`, untrusted chunks are judged on a bounded thread pool through `submit_in_context`. Output order is preserved, and a chunk whose verdict can't be parsed is still kept, as before.
- With the default of 1, the loop is sequential exactly as before.

---

## 3. The new `web_deploy/` folder

It sits outside the feature-first layout, like `deploy/`, with its own venvs. None of its dependencies were added to the root `requirements.txt`. Layout and roles are in `web_deploy/README.md`; the full guide is in `docs/web_deploy.md`.

### 3.1 `common/` (standard library only)
- **`tenancy.py`:**
  - Firm slug rules: 3–32 characters, lowercase, letters, digits and single hyphens, plus a reserved list.
  - `database_name()` gives `onyx_firm_<slug>`.
  - `firm_root()`.
  - `slug_from_host()`, which only accepts one subdomain level under the base domain.
- **`signing.py`:**
  - The gateway signs requests to the box: an HMAC-SHA256 over method, path, query, firm, user, role and timestamp, valid for 60 seconds.
  - **The body is not signed,** so uploads can stream. A captured request could therefore be replayed within those 60 seconds.
  - `sign_service` / `verify_service` sign box-internal GPU handoff calls (local mode).

### 3.2 `api/`: the box API (Starlette, one process)
- **`engine_port.py`:**
  - The `IEngineHost` port.
  - `ApplianceIngesting`.
  - `LiveGate`, the pause/resume admission counter for local mode.
- **`engine_host.py`** is the only module that imports `app.*`:
  - At startup it builds one `ModelKit` and warms it, after waiting up to 10 minutes for llama-server's health.
  - It creates each firm's database (`firm_db.ensure_firm_database`), vault, reports and uploads directories, and `IzvoriaEngine` on first use, with a small connection pool per firm (3 + 3).
  - It implements `pause()` and `resume()` (local mode).
- **`app.py`** routes:
  - chat and reasoned chat, search, evidence;
  - conversation and search history;
  - reports: plan, get, markdown, run, resume, cancel;
  - library, sources (with Range), uploads, the ingestion queue;
  - `/health`, unsigned and returning counts only;
  - `/internal/gpu/{pause,resume}`: service-signed, loopback only.

  Every model-using call runs inside `host.live()`.
- **`runs.py`:** long calls become runs. Server-Sent Events carry progress, then one `result` or `error`. They resume from `Last-Event-ID`, a run is visible only to its owner, and finished runs are kept for 30 minutes. Runs live in memory, so an API restart loses the live view, not the work.
- **`runner.py`:** separate live and report thread pools. Each run enters the request's language scope and progress routing.
- **`progress.py`:** a ContextVar that routes JobManager ticks to the right run.
- **`mapping.py`:** engine objects to JSON, adapted from the desktop's `engine_gateway`. **Never** includes file paths, only `has_source`.
- **`sources.py`:** a cited file is re-resolved by chunk id and must lie inside the firm's own vault. Safe-serving headers.
- **`config.py`**, **`firm_db.py`**.

### 3.3 `ingest_worker/`
- **`queue.py`:** FIFO `ingest_jobs` in a control database (Postgres `onyx_control` on the box, SQLite locally), created with `create_all` since it is ops state, not appliance schema.
  - A claim is a conditional update, so no job can be claimed twice.
  - Jobs interrupted by a shutdown are **re-queued**, not failed.
  - Firm listings never show box paths.
- **`supervisor.py`:**
  - Runs each job in a fresh child process, with the firm's database, vault and reports directory, the batch's language, the in-process backend, keep-warm off and VRAM headroom.
  - A child that dies without recording a result fails its job.
- **`run_job.py`:** the child. It creates the firm's database if needed, builds the appliance's own engine, ingests, and writes progress (throttled) and the report.
- **`gpu_arbiter.py`:** local mode (section 4).

### 3.4 `gateway/` (Hetzner VPS)
- **`store.py`:** SQLite tables for firms, users and sessions.
  - Sessions are server-side and stored as a hash of the cookie token.
  - Disabling a user, resetting a password or disabling a firm signs them out everywhere.
- **`passwords.py`:** stdlib scrypt (n=2¹⁴, r=8, p=1), minimum 10 characters, with constant-ish timing for unknown accounts.
- **`app.py`:**
  - Firm by host; the session must belong to the host's firm.
  - The cookie is host-only and HttpOnly, SameSite=Lax, and Secure when HTTPS.
  - CSRF protection checks the `Origin` header on writes.
  - Logins are rate-limited in memory: 8 failures per 15 minutes per IP and per account.
  - The owner console's API sits under `/admin/api/*`.
  - `/internal/tls-ask` lets Caddy check a firm exists before issuing its certificate.
  - The SPA is served as a fallback route.
- **`proxy.py`:** streams requests both ways with allowlisted headers. Cookies never reach the box.
- **`box.py`:** Vast start, wait, health and stop. It reuses `deploy/gateway/vast_client.py`. `LocalBox` is for local mode.
- **`idle.py`:** stops the box only when idle *and* the box reports nothing in flight, running or queued.
- **`admin_cli.py`:** create-owner, create-firm, enable/disable-firm, create-user, reset-password, disable/enable-user, list. Passwords are prompted, or taken from `ONYX_ADMIN_PASSWORD`.
- **`config.py`**.

### 3.5 `frontend/` (React 18 + Vite 5 + TypeScript)
- **Views:**
  - Login, plus a firm field in login-routing mode;
  - "Waking the appliance";
  - Chat: chat or reasoned mode, a per-question language picker, clickable `[Doc N]` chips, evidence cards, a source viewer for video, audio, PDF page, highlighted text lines and images;
  - Search: standard or verified, with history;
  - Reports: topic, editable outline, progress bar, Stop, Resume, clickable citations, Save as Markdown;
  - Ingest: upload with progress, the queue with your position;
  - Library, Account (change password), the owner's Admin console.
- **Language:** the UI is in English and Serbian. The display language is remembered in `localStorage`.
- **Banner:** shown while ingestion has the GPU (local mode).
- **Build:** writes to `web_deploy/gateway/static`, which is gitignored.

### 3.6 `provision/`
- **Box:**
  - `provision.sh`: Postgres 15 + pgvector, the `vector` extension in `template1`, a `CREATEDB` role, the `onyx_control` database, a self-signed TLS certificate, secrets, the venv and the llama-server build.
  - `build_llama_server.sh` (CUDA, **static** since local mode), `llama_server.sh`, `supervisord.conf`, `onstart.sh`, `sync_to_box.sh`, `api.env.example`.
- **Gateway:** `install_gateway.sh`, `Caddyfile` (on-demand TLS), `onyx-web-gateway.service`, `gateway.env.example`.
- **`seed_demo.py`:** copies a folder of sample files into a demo firm's uploads and queues them.

### 3.7 Reuse of `deploy/` code

These are imported, not copied, and none were modified:
- `deploy.gateway.vast_client`
- `deploy.instance.agent.sources`: vault-path check, file kinds
- `deploy.instance.agent.media`: ffmpeg preparation and cache
- `deploy.instance.agent.uploads`: filename sanitising

`deploy/tests` keeps covering them (234 pass). If `deploy/` is ever deleted, `web_deploy/` breaks.

---

## 4. Local dev mode (second request)

Three settings in `web_deploy/local/local.env` select it. `local.env` is gitignored and holds a generated secret.

| Setting | Effect |
|---|---|
| `BOX_MODE=local` | The gateway's box is `LocalBox` (always running). No Vast, no idle monitor, and login doesn't "wake" anything. |
| `FIRM_ROUTING=login` | One LAN address. The firm is typed at sign-in (`admin` for the owner). `/auth/me` advertises it. |
| `GPU_SHARING=exclusive` | The ingestion worker takes the whole card. |

**How the GPU handoff works** (`ingest_worker/gpu_arbiter.py`, `api/engine_port.LiveGate`, `engine_host.pause/resume`):
1. When a job is claimed, the worker calls `POST /internal/gpu/pause`.
2. The API stops admitting chat, search and report work (409, kind `ingesting`), waits for admitted work to finish, including a running report, and frees its pinned models.
3. The worker stops `llama-server`, which it owns in this mode, and runs the queue.
4. When the queue is empty, it starts `llama-server`, waits for it to be healthy, and calls `/resume`.

History, evidence, sources and the library stay readable throughout. The GPU changes hands once per queue, not per batch.

**`run_local.py`:**
- **Commands:** `setup`, `build-llama-server`, `build-web`, `admin …` (wraps the admin CLI), and the default `start`.
- **What `start` runs:** the API on `127.0.0.1:8095`, the ingestion worker, and the gateway on `0.0.0.0:8000` over **plain HTTP**.
- **Logs and data:** under `onyx_data/local-web/`.
- **Ctrl+C** stops everything, `llama-server` included. This was verified.

**Also added for local mode:**
- the API waits for `llama-server`'s health before warming up;
- `/health` reports `paused`, which the gateway passes on as `BoxStatus.paused` and the frontend shows as a banner.

---

## 5. Bugs found by running it for real

| # | Found by | Problem | Fix |
|---|---|---|---|
| 5.1 | llama-server spike | `response_format: json_object` / `json_schema` fails on Qwen3.5 with "Failed to initialize samplers" (llama.cpp `2e7c58c`). It would have broken the scrubber verdict, the critique, the reasoner and report outlining. | The client sends llama.cpp's `json.gbnf` as a raw `grammar` instead (`JSON_GRAMMAR`). Re-check when bumping llama.cpp. |
| 5.2 | End-to-end run | Pinned warm-up was refused with `OutOfVramError`: admission used BGE-M3's on-disk size, 4.3 GB against 1.1 GB real. | `warm_up` admits by `required_vram_mb` (2.3). |
| 5.3 | Gateway tests | The proxy used `aiter_raw()`, which fails on responses already read into memory. | `aiter_bytes()`, dropping `content-length` if the upstream sent content-encoding. |
| 5.4 | Gateway tests | `/admin/api/firms/{slug}/{action}` shadowed `/admin/api/firms/{slug}/users`. | Reordered, and `action` validated. |
| 5.5 | Real run | The frontend's progress phase names didn't match the engine's `PipelinePhase` values. | Corrected, with Serbian translations. |
| 5.6 | Local setup | The `llama-server` build linked shared libraries by absolute path, so moving or cleaning the build tree breaks it. | `-DBUILD_SHARED_LIBS=OFF`: one 62 MB self-contained binary. |
| 5.7 | Local run | The status showed "booting" for a moment after every local login. | Login only wakes a Vast box. |
| 5.8 | Writing provision.sh | A nested heredoc truncated the script. | Repaired and syntax-checked with `bash -n`. |

---

## 6. Docs, memory and config

### 6.1 `CLAUDE.md`
Four additions; the file also carries your earlier uncommitted edits:
- **Model management:** "Pinned and external types".
- **Languages:** "The hosted API overrides it per request".
- **Known gotchas:** the mutual-exclusion bullet now notes how hosted mode lifts it, plus a new bullet, "Live phases keep their preload per thread".
- **Where things stand:** a "Hosted, multi-firm web version" paragraph, including local mode.

### 6.2 Other docs and config
- **`CHANGELOG.md`:** a new top entry, "Hosted, multi-firm web version (`web_deploy/`)", including local mode.
- **`docs/web_deploy.md` (new):** architecture, the `app/` settings table, VRAM table, M0 spike runbook with the partial 3060 results, setup for box and gateway, demo firms, local dev mode, tests, open items.
- **`web_deploy/README.md` (new).**
- **`pytest.ini`:** `web_deploy/tests` added to `testpaths`. It skips itself under the appliance venv when the web stack is missing.
- **`.gitignore`:** `/web_deploy/.venv/`, `/web_deploy/frontend/node_modules/`, `/web_deploy/gateway/static/`, `/web_deploy/local/local.env`.

### 6.3 Claude memory
`~/.claude/projects/-home-aleksa-code-onyx-v3/memory/hosted-multi-user-direction.md` was rewritten to record the decisions, the build, the measurements and local mode. Its index line in `MEMORY.md` was updated.

---

## 7. Side effects on your machine

| What | Where | Status |
|---|---|---|
| `starlette` 0.52.1 and `uvicorn[standard]` 0.54.0 (with their deps) installed into the **appliance venv** | `venv/` | Kept. Needed for local mode. Not in `requirements.txt`. |
| Test venv for `web_deploy/tests` | `web_deploy/.venv/` | Kept, gitignored |
| npm install (276 packages) and `package-lock.json` | `web_deploy/frontend/` | Kept. `node_modules` is gitignored; the lockfile should be committed. |
| Built web app | `web_deploy/gateway/static/` | Kept, gitignored |
| llama.cpp clone and **static** `llama-server` build (commit `2e7c58c`) | `onyx_data/local-web/llama.cpp/` | Kept for local mode |
| Generated local config with a secret | `web_deploy/local/local.env` | Kept, gitignored, mode 600 |
| Scratch llama.cpp build, test scripts | the session's `/tmp/claude-1000/...` scratchpad | Temporary |
| Database `onyx_alembic_scratch` | `db_test` | Created and **dropped** |
| Database `onyx_firm_spike_firm` | `db_test` | Created and **dropped** |
| Database `onyx_firm_test_local` | `db_dev` | Created and **dropped** |
| Test owner, firm and users; test uploads, logs, queue | `onyx_data/local-web/` | **Deleted** |
| GPU processes (`llama-server`, API, worker, gateway) | – | All stopped; VRAM back to its ~0.7 GB baseline |

---

## 8. Tests: what exists and what was run

**New and changed tests:**
- **`app/` changes:**
  - `tests/core/model_manager/data/test_llama_server_chat_model.py` (17)
  - `test_headroom_vram_monitor.py` (4)
  - orchestrator: pinned, external and strict residency, plus warm-up admission (11)
  - `test_llama_cpp_chat.py` (backend dispatch)
  - `tests/orchestration/utils/test_preload_slot_unit.py`
  - engine: concurrent mode, owner and report cap (9)
  - bootstrap: model kit (4)
  - language scope (6)
  - scrubber adapter and concurrent scrubbing
  - session memory and search history: owner
  - updated exact-argument assertions (`owner_id=None`) across engine, session memory and search history tests
- **Integration:**
  - owner isolation for session memory and search history;
  - the pre-Alembic migration test now drops the new `owner_id` columns too.
- **`web_deploy/tests` (169):** signing, tenancy, runs, progress, runner, mapping, the API app with a fake engine, the GPU handoff, queue, supervisor, GPU arbiter, gateway store, CLI, box, idle, the gateway app, local mode, the launcher, seeding.
- **Frontend (Vitest, 10):** citation chips, evidence cards, language picker, the run event client, the report's clickable citations, the login firm field.

**Last results** (all green):

| Suite | Result |
|---|---|
| `pytest tests` | 2,217 passed |
| `QT_QPA_PLATFORM=offscreen pytest frontend/tests` | 364 passed (one timing-sensitive desktop test, `test_engine_bridge_unit`, failed once while a CUDA build loaded the CPU, and passed alone and on rerun; I did not touch that code) |
| `web_deploy/.venv/bin/pytest web_deploy/tests` | 169 passed |
| `deploy/.venv/bin/pytest deploy/tests` | 234 passed |
| Vitest + `tsc --noEmit` | 10 passed, typecheck clean |
| Integration: report load-count (`test_deep_research_report_pipeline_integration.py`) | 7 passed |
| Integration: warm residency | 3 passed |
| Integration: engine | 1 passed |
| Integration: database, session memory, search history | all passed |

**Not run:**
- the whole `integration_tests` suite in one go (only the subsets above);
- `integration_tests/core/model_manager/test_orchestrator_race_condition.py`, which was deselected by `-m integration` and not rerun without the marker.

---

## 9. Measurements

All on the RTX 3060 (12 GB) with llama.cpp `2e7c58c` and Qwen3.5-9B Q4_K_M:

| Measure | Result |
|---|---|
| Qwen3.5 hybrid layers with `--parallel N --kv-unified` | Works. Each slot may use the whole context (`n_ctx_slot` = total). |
| llama-server VRAM, 64K context, 4 slots | ~6.6 GB |
| llama-server VRAM, 128K context, 2 or 3 slots | ~8.1 GB (the slot count barely matters) |
| Plus BGE-M3 and reranker under a heavy embed and rerank | 0.71 GB of 12 GB left free |
| `/tokenize` against llama-cpp-python | Identical, including Serbian diacritics, Cyrillic and special tokens |
| 4 chat requests (~220 tokens each), one after another vs at once | 16.8 s vs 7.1 s (50 → 118 tokens/s aggregate) |
| Scrubber on the 9B, 8 chunks, concurrency 1 / 3 / 4 | 7.3 s / 4.4 s / 4.4 s, same verdicts, no parse failures |
| End to end through `EngineHost`: 4 users (2 en, 2 sr) at once | All answered within 8.2 s, correctly cited, in the right language |
| Local mode over the LAN: 2 users at once after an ingestion | 5.5 s |
| Document ingestion through queue and child worker | 27 s for one small TXT, including process start |

---

## 10. Where the code differs from the approved plan

| Plan said | Built | Why |
|---|---|---|
| argon2-cffi for passwords | stdlib `hashlib.scrypt` | No native dependency on the gateway |
| Caddy wildcard certificate via DNS-01 (custom Caddy build) | Caddy **on-demand TLS** with an `ask` endpoint | Stock Caddy, no DNS API keys; only real firms get certificates |
| `sse-starlette` | Hand-written SSE | One less dependency |
| Copy `deploy/` modules | **Import** them | Less duplicated code (3.7) |
| An `IChatModel` port with mechanical adapter edits | Port added, but the three Qwen adapters were not edited; the server client duck-types `tokenize` / `n_ctx` | Fewer changes |
| `owner_id` threaded through the report pipeline | `set_owner()` after `propose_plan` | Kept the pipeline signature unchanged |
| M6: Playwright end-to-end tests | **Not done**; Vitest component tests only | – |
| M7: provisioning tested on a fresh Vast box | **Not done** (no box) | Needs your Vast and Hetzner accounts |
| 4 slots at 192K on 24 GB | Hosted defaults unchanged; local uses 3 slots at 128K, measured | – |
| (not in plan) local dev mode | Added at your request | Section 4 |

---

## 11. Known gaps and risks: what to review first

Roughly in order of importance.

1. **The hosted deploy has never run.** `provision.sh`, `install_gateway.sh`, the Caddy on-demand TLS, Vast start/stop, the pinned box certificate and `supervisord.conf` are untested on real machines. Expect a round of fixes the first time.
2. **3090 spike items are open:**
   - VRAM at 192K with 4 slots;
   - what happens when the shared KV cache runs out (`KvTokenBudget` is an *estimate*: tokenized message text plus 8 tokens per message plus `max_tokens`);
   - a ~120K-token request beside short ones.
3. **Scrubber quality with the 9B is unmeasured.** In the toy test it kept 6 of 8 chunks for a question only one chunk answers, which is lenient. Run `scripts/eval/rag_eval.py` against the 7B before relying on it. The appliance still uses the 7B by default.
4. **Two real reports have never run at the same time.** That path has only been tested with fakes. I checked that no pipeline, phase, feature API or `JobManager` keeps per-run state on itself.
5. **Security has had no independent review.** Hand-written parts:
   - sessions, CSRF via `Origin`, the in-memory rate limit (it resets when the gateway restarts);
   - the HMAC scheme, with an unsigned body and a 60 s replay window;
   - the vault-path check (reused from `deploy/`).

   Local mode is plain HTTP on the LAN. Consider `/code-review` or `/security-review` on this diff.
6. **Hosted data handling.** No off-box backups of firm databases and vaults. "No data leaves the premises" doesn't apply to the hosted version, so it needs its own statement.
7. **Migration 0005 runs on every database at its next start,** including real appliances. It only adds nullable columns and indexes, but it is a schema change.
8. **Appliance-visible code paths changed even with default settings:**
   - the scrubber builds and calls its model through the shared helpers, and strips `<think>` (2.2);
   - `require_idle()` now also counts in-flight calls (2.5);
   - phases use `PreloadSlot` (2.4);
   - embed and rerank take a lock (2.4).

   Tests show the same behaviour, but these are the lines to read.
9. **The frontend is the least tested layer:** 10 component tests and no browser end-to-end tests. Its local mode was exercised over HTTP, not by clicking through it in a browser.
10. **Coupling.** `web_deploy/` depends on `deploy/` modules (3.7). If `deploy/` is retired, move those four modules first.
11. **`LLAMA_CPP_REF` defaults to `master`.** Pin it, ideally to `2e7c58c`, the build everything here was measured with. The JSON grammar workaround (5.1) is tied to that behaviour.

---

## 12. How to undo it

Nothing is committed. Your earlier uncommitted work (`deploy/`, `frontend/`, two docs, part of `CLAUDE.md`) shares the working tree, so **don't run a blanket `git checkout .`**.

**To drop this session's work only:**
```bash
git checkout -- app tests integration_tests pytest.ini .gitignore CHANGELOG.md   # session-only changes
git checkout -p -- CLAUDE.md          # keep your own hunks, drop the four listed in 6.1
rm -rf web_deploy docs/web_deploy.md docs/session_changelog_2026-10-04.md
rm app/core/database/data/migrations/versions/0005_history_owner.py \
   app/core/model_manager/data/headroom_vram_monitor.py \
   app/core/model_manager/data/llama_server_chat_model.py \
   app/core/model_manager/exceptions/server.py \
   app/orchestration/utils/preload_slot.py \
   tests/core/model_manager/data/test_headroom_vram_monitor.py \
   tests/core/model_manager/data/test_llama_server_chat_model.py \
   tests/orchestration/utils/test_preload_slot_unit.py
rm -rf onyx_data/local-web                                  # the llama-server build and local data
venv/bin/pip uninstall starlette uvicorn                    # optional
```

**If a database already ran migration 0005,** its `alembic_version` is `0005`, and the reverted code has no revision `0005`. Either keep the migration file, or drop the three `owner_id` columns and set `alembic_version` back to `0004` by hand. Migrations are forward-only (`docs/core/database.md`).

**To commit it instead,** review sections 2 and 11, then commit `app/`, `tests/`, `integration_tests/` and the migration separately from `web_deploy/`. The appliance-side changes stand on their own, and are what a regression would most likely trace back to.
