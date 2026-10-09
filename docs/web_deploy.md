# Hosted Onyx for many firms (`web_deploy/`)

This is the hosted web version that demos are shown on and that users use. It runs on two machines:

- **A Hetzner VPS.** Always on. It handles firms, users and sign-in, and wakes the GPU box.
- **A rented Vast.ai GPU box.** On demand. It runs the appliance itself.

Everyone signs in at one address. Each firm has its own sign-in name and shared password, its own database and its own file vault, shared by all of that firm's users. Several users can chat, search and run reports at the same time. Ingestion waits its turn in a queue.

It sits beside the single-tester noVNC demo (`deploy/`, `docs/deploy.md`) and replaces it. It also reuses four of the demo's modules:

- the Vast client;
- the source vault check;
- the ffmpeg media preparer;
- upload filename sanitising.

It is ops tooling, not the product's security model. The appliance proper stays on-premise and air-gapped, and nothing in `web_deploy/` is installed there.

## How a question travels

```
browser ──https──▶ Caddy (VPS) ──▶ gateway (VPS)                      ──https, signed──▶ API (GPU box)
                   one host, TLS     firms, users, sessions (SQLite)                      one process, all firms
                                     Vast start/stop, idle stop                            │
                                                                                          ├─ firm engine (own DB, vault)
                                                                                          ├─ firm engine …
                                                                                          └─ shared model kit
                                                                                               ├─ BGE-M3 + reranker (pinned)
                                                                                               └─ llama-server: Qwen3.5-9B,
                                                                                                  parallel slots
                                                                     ingestion supervisor ─▶ one child process per job
```

1. **The gateway identifies the firm and the user.**
   - The session names the user, and the user's account names the firm (see "Signing in" below).
   - The firm and the account are re-read on every request, so disabling either takes effect at once.
   - The gateway forwards `/api/*` to the box with an HMAC over method, path, query, firm, user, role and timestamp (`web_deploy/common/signing.py`).
   - The box never sees a cookie or a password.
2. **The API checks the signature and picks the firm's engine.**
   - It hands that firm's engine to a worker thread.
   - A firm's engine is an ordinary `OnyxEngine` built by `OnyxBootstrapper.build(database=…, vault_dir=…, reports_dir=…, kit=…)`. Every firm shares one `ModelKit`, so a new firm costs a connection pool, not a model load.
3. **Long calls become runs.**
   - A chat turn, search, report scoping or report run returns a run id at once (`web_deploy/api/runs.py`).
   - The browser follows the run's Server-Sent Events: `progress` ticks, then one `result` or `error`.
   - A reconnect resumes from `Last-Event-ID`.
4. **Everything is scoped to the user.**
   - History is per user: `owner_id` on `session_turns`, `search_history` and `research_reports` (migration 0005).
   - Each user sees only their own conversations, searches and reports.
   - A cited source is re-resolved by chunk id and must lie inside the firm's own vault.

## Signing in, accounts and requests

**Two steps, one address** (`web_deploy/gateway/app.py`, `store.py`):
1. **The firm.** Its sign-in name (the slug) and a password everyone at the firm shares (`POST /auth/firm`). This gives the browser a firm pass: the `onyx_firm` cookie (HttpOnly, SameSite=Strict) for 20 minutes, or 30 days if "Remember this firm on this device" is ticked.
   - A wrong password, an unknown firm, a disabled firm and a firm without a password all get the same answer and cost the same one scrypt.
   - Failures are limited to 8 per address and 30 per firm per 15 minutes. The per-firm limit is higher because the password is shared, so a stranger could otherwise lock a whole firm out.
2. **The person.** Their own email and password (`POST /auth/login`), within the firm the pass is for. A firm in the request body is never trusted. This gives the 14-day `onyx_session` cookie, as before.

The owner's console is the firm `admin`, with its own shared password (`set-firm-password admin`).

**Signing out and switching:**
- Signing out keeps the firm pass, so the next person on a shared office machine starts at their own account.
- "Switch firm" (`/auth/firm/forget`) clears both cookies.
- Changing a firm's password ends every device's pass, so each must enter the new one, but nobody who is already signed in is signed out. To cut everyone off, disable the firm.

**Who manages accounts:**
- **The owner** (console, or `admin_cli`) creates firms, each with a firm password, and each firm's first administrator.
- **A firm's administrators** (role `admin`, the web app's Accounts page, `/firm/api/*`) add, disable and re-enable accounts, reset passwords, make other people administrators, and change the firm password. Their own firm only: anyone else's user id is a 404, and owners can't be touched.
- An administrator can't disable or demote themselves, and a firm always keeps one enabled administrator.

**Public pages:**
- **Request Onyx for your firm** (`/request-access`, `POST /public/signup`). Contact details plus the firm profile from `docs/glossary_authoring/firm_profile_template.md`. The emailed copy follows the template's sections, so it pastes straight into the glossary creator prompt as `{{FIRM_PROFILE}}`. The names section is optional and warns that the form travels by email.
- **Help & support** (`/help`, `POST /public/support`). Linked from both sign-in steps and the request page.

How the public pages are handled:
- **Storage.** Both are stored in the gateway database (`inbound_requests`) and listed under Requests in the owner's console, which shows mail status, a resend, and handled/spam.
- **Email.** They are emailed by `gateway/mailer.py`: plain SMTP, on a background thread, with Reply-To set to the requester.
- **Abuse protection.** Same-origin only, a 32/16 KB body cap, field limits, a hidden honeypot field and a minimum fill time (a bot is told it worked and nothing is kept), 5 per address per hour, and at most 100 emails a day.

**Email settings** (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_STARTTLS`/`SMTP_SSL`, `MAIL_FROM`, `SIGNUP_TO`, `SUPPORT_TO`):
- Mail is off when there is no `SMTP_HOST`, or when there is a user without a password: requests are then only stored.
- Hetzner Cloud blocks outbound 25 and 465 on new projects, so use 587 with STARTTLS.
- Gmail (dev) needs 2-step verification and an app password, and `MAIL_FROM` must be the same address.

**Look.** The web app copies the desktop app (`frontend/onyx_desktop/ui/theme.py`):
- the ink sidebar, serif headings, warm neutrals and the deep green accent;
- dark by default, with a light toggle;
- three panes for Chat, Search and Report, with Source review on the right.

Fonts are self-hosted (`@fontsource`), so no page asks a third party for anything.

## What changed in `app/` (all off by default; the desktop is untouched)

| Setting | Appliance | Hosted | What it does |
|---|---|---|---|
| `LLM_BACKEND` | `llama_cpp` | `llama_server` | Points the Qwen chat adapters at a local `llama-server`. `build_llama()` / `chat_completion()` dispatch to it; the client is `LlamaServerChatModel`. |
| `MODEL_PINNED_TYPES` | `[]` | embedder, reranker | Loaded once (`ModelKit.warm_up`) and never released, evicted or unloaded. |
| `MODEL_EXTERNAL_TYPES` | `[]` | the four Qwen roles | Served by llama-server: a client per role, no VRAM budgeted, never unloaded. |
| `ENGINE_CONCURRENT_LIVE` | false | true | See below. |
| `REPORT_MAX_CONCURRENT` | 1 | 2 | How many reports may run at once. |
| `RETRIEVAL_SCRUBBER_CONCURRENCY` | 1 | 3 | Chunks judged in parallel slots, instead of one after another. |
| `VRAM_HEADROOM_MB` | 0 | (ingestion child) 1500 | The ingestion worker admits its loads against the card minus the live side's growth. |
| `VIDEO_CLIPS_ENABLED` | true | false | No per-chunk clips; see "Storage". |
| `KEEP_EXTRACTED_AUDIO` | true | false | A recording's WAV is deleted once it is indexed. |
| `COMPACT_MEDIA_ORIGINALS` | false | true | An indexed recording's original becomes a 720p H.264/AAC copy. |

**What `ENGINE_CONCURRENT_LIVE=true` changes:**
- Live calls run concurrently and are counted (`busy_snapshot()`).
- Reports run beside them, up to the cap.
- The orchestrator refuses any live model type that is neither pinned nor external (`ResidencyError`).

**Why pinned and external, not a refcount.** `release_model` is deliberately not a refcount (CLAUDE.md). Two requests sharing a model could free it under each other: llama.cpp aborts inside `free_model`. A model nobody can ever free is safe to share.

**The remaining shared state was made per request:**
- **Phase preloads** use `PreloadSlot`, one per thread, so one request can't resolve another's preload.
- **The embedder and reranker** take a per-adapter lock around `encode` and `predict`, because the HF fast tokenizer is not thread-safe.
- **The answer language** is a ContextVar override: `language_scope(code)`, `active_language()`. The API sets it per request inside the worker thread. `submit_in_context` carries it into thread pools.

## VRAM on one 24 GB card (plan estimates; replace with the spike's measurements)

| Item | VRAM |
|---|---|
| CUDA contexts (llama-server, API, ingestion child) | ~1.0 GB |
| Qwen3.5-9B Q4_K_M weights (llama-server) | 5.7 GB |
| Unified q8_0 KV cache, 192K tokens across 4 slots | ~3.0 GB |
| DeltaNet recurrent state ×4 and compute buffers | ~1.0 GB |
| BGE-M3 fp16 (measured) | 1.09 GB |
| bge-reranker-v2-m3 (measured) | 1.09 GB |
| Embed/rerank activations | ~0.8 GB |
| **Live total** | **~13 GB** |
| **Ingestion (the rest)** | **~10–11 GB** |

Qwen2-VL (7.5 GB), the largest ingestion model, fits in the ingestion share.

If the live side measures over 13.5 GB:
- Shrink `LLM_SERVER_KV_TOKENS` before shrinking `LLM_SERVER_PARALLEL`.
- Never shrink a single request's 128K context.

## M0 spike: run this before anything else (rented 3090)

1. Build `llama-server` (`web_deploy/provision/build_llama_server.sh`). Start it the way `llama_server.sh` does: `--parallel 4 --kv-unified -c 196608 -ctk q8_0 -ctv q8_0 -fa on --jinja --reasoning-format none`.
2. Confirm Qwen3.5's hybrid layers load with parallel slots, and that `/props` reports the slot `n_ctx`.
3. Measure VRAM (`nvidia-smi`) at 1, 2, 4 and 6 slots and at 128K, 160K, 192K and 256K total. Pick `LLM_SERVER_PARALLEL` and `LLM_SERVER_KV_TOKENS`.
4. Send four requests at once, one of them ~120K tokens. Watch what happens when the KV cache runs out: an error, or a retry. This decides how strict `KvTokenBudget` must be.
5. Measure per-request and aggregate tokens/s, against the in-process baseline.
6. Run `scripts/eval/rag_eval.py` with the 9B as scrubber: `LLM_BACKEND=llama_server RETRIEVAL_SCRUBBER_MODEL_PATH=models/Qwen3.5-9B-Q4_K_M.gguf RETRIEVAL_SCRUBBER_CONCURRENCY=3`. Compare keep/drop decisions and latency with the 7B. The target is 8 chunks in under 8 × 2.4 s.
7. Record the numbers here. `build_llama_server.sh` is pinned to `b11396` (`2e7c58c`), the release the dev-card run below validated; if the 3090 needs another release, re-run these checks on it and change the pin.

**If parallel slots plus unified KV fail on Qwen3.5, stop: the fallback (`--parallel 2` without `--kv-unified`) caps a request's context below 128K, and that needs a decision.**

### What a partial run on the 12 GB dev card showed (2026-10-04, llama.cpp `2e7c58c`)

**Setup:** `--parallel 4 --kv-unified -c 65536 -ctk/-ctv q8_0 -fa on --jinja --reasoning-format none`.

**What works:**
- Qwen3.5-9B's hybrid layers load with parallel slots. The server reports `n_slots = 4, n_ctx_slot = 65536, kv_unified = true`: any one slot may use the whole cache.
- The whole server takes **~6.6 GB** at 64K.
- `/tokenize` matches `llama_cpp.Llama.tokenize(special=True)` exactly, including Serbian diacritics, Cyrillic and special tokens.
- `chat_template_kwargs.enable_thinking=false` gives an empty `<think></think>`, which `strip_reasoning` removes.

**One fix it forced:**
- `response_format: json_object` (and `json_schema`) fails with "Failed to initialize samplers" on Qwen3.5 under `--jinja`.
- A raw GBNF grammar works. `LlamaServerChatModel` therefore sends llama.cpp's `json.gbnf` instead (`JSON_GRAMMAR`).
- Re-check this when bumping `LLAMA_CPP_REF`.

**Throughput:**
- 4 chat requests (~220 tokens each) took **16.8 s** one after another and **7.1 s** at once: 50 → 118 tokens/s aggregate.
- The retrieval scrubber on the 9B judged 8 chunks in 7.3 s sequentially and in **4.4 s** with `RETRIEVAL_SCRUBBER_CONCURRENCY=3`, with the same verdicts and no parse failures. In-process it was 5.3 s per chunk.

**End to end (EngineHost on `db_test`):**
- A document was ingested through the queue and a child worker.
- Four users (2 en, 2 sr) asked at once and got correctly cited answers in their own languages, all within 8.2 s.
- Each user saw only their own history, and the shared models stayed resident throughout.
- This run is what found that a pinned model must be admitted by its declared requirement, not its on-disk size (see `ModelManagerOrchestrator.warm_up`).

**Still open, for the 3090:**
- VRAM at 192K and up;
- what happens when the KV cache is exhausted;
- scrubber keep/drop quality on `rag_eval` against the 7B;
- a ~120K-token request beside short ones.

## Setting it up

**Settings made ahead of time (optional).** `web_deploy/provision/prod/` (gitignored, never synced) can hold the two production files, made from the examples with one shared `WEB_API_SECRET` and a fresh database password:
- `api.env` → `/workspace/web/api.env` on the GPU box, copied **before** `provision.sh` (which then keeps it instead of generating secrets);
- `gateway.env` → `/etc/onyx-web-gateway.env` on the VPS (`chmod 600`), copied **before** `install_gateway.sh` (which then keeps it).

Without them, both scripts generate the files and you copy `WEB_API_SECRET` across by hand. Never copy `prod/` to either machine as part of the repo: the gateway file holds the Vast API key.

### GPU box (Vast.ai)

Template:
- an nvidia/cuda `*-cudnn-devel-ubuntu22.04` image;
- one 24 GB GPU;
- port **8095** exposed;
- on-start script `bash /workspace/onyx_v3/web_deploy/provision/onstart.sh`.

Steps:
1. Build the web app on the dev box:
   ```bash
   cd web_deploy/frontend && npm ci && npm run build   # -> web_deploy/gateway/static
   ```
2. Copy the code and models over:
   ```bash
   web_deploy/provision/sync_to_box.sh root@<ip> <ssh port>
   ```
3. Provision the box:
   ```bash
   ssh … 'bash /workspace/onyx_v3/web_deploy/provision/provision.sh'
   ```
   This installs Postgres 15 with pgvector (in `template1`, so each firm database has it), builds `llama-server`, writes `/workspace/web/api.env` with fresh secrets, and makes a self-signed certificate for the API.
4. Note what it prints: `WEB_API_SECRET`, and the path of the API certificate.

### Gateway (Hetzner VPS)

1. DNS: one A record for `onyx.example.com`, pointing at the VPS.
2. Copy the repo, with the built `web_deploy/gateway/static`, to `/opt/onyx_v3`. Then:
   ```bash
   sudo bash /opt/onyx_v3/web_deploy/provision/install_gateway.sh onyx.example.com
   ```
3. Fill in `/etc/onyx-web-gateway.env`:
   - `WEB_API_SECRET`;
   - `VAST_API_KEY` and `VAST_INSTANCE_ID`;
   - the box certificate at `/etc/onyx-web/box-cert.pem`;
   - `SMTP_*`, `MAIL_FROM`, `SIGNUP_TO` and `SUPPORT_TO`, for the request and help pages.

   Then `systemctl restart onyx-web-gateway`.
4. Create your owner account (it also asks for the console's firm password):
   ```bash
   cd /opt/onyx_v3 && sudo -u onyx GATEWAY_DB=/var/lib/onyx-web/gateway.sqlite3 \
       /opt/onyx-web-gateway/bin/python -m web_deploy.gateway.admin_cli create-owner you@example.com
   ```
5. Sign in at `https://onyx.example.com` as the firm `admin`, then with your email. Create firms, each with a firm password, and their first administrators. `admin_cli` does the same from the shell (`create-firm … --generate` makes up the firm password and prints it once).

Caddy gets and renews the one certificate for that host automatically.

**Upgrading a deployment from subdomains.** Firms used to live at `<firm>.<domain>`.
1. Deploy the code. The gateway database gains its new columns at startup, and existing firms have no firm password, so nobody can sign in yet.
2. Run `admin_cli set-firm-password <slug>` for every firm (`list` flags the ones missing), and `set-firm-password admin` for the console.
3. Install the new Caddyfile (`install_gateway.sh` does it) and replace `BASE_DOMAIN` with `PUBLIC_URL` (an old `BASE_DOMAIN` still works, as `https://<BASE_DOMAIN>`). The wildcard DNS record can go.
4. Tell each firm its sign-in name and password. Everyone signs in again, because the old cookies belonged to the subdomains.

### Demo firms

Create the firm on the gateway (`legal-demo`, "Demo"), then queue sample material on the box:
```bash
cd /workspace/onyx_v3 && set -a && . /workspace/web/api.env && set +a
venv/bin/python -m web_deploy.provision.seed_demo legal-demo document en /workspace/samples/contracts
venv/bin/python -m web_deploy.provision.seed_demo clinic-demo audio sr /workspace/samples/consultations
```

## Local testing on dev hardware (one 12 GB GPU, your LAN)

The same code runs on the development machine for personal testing (`web_deploy/local/`). Sign-in works the same way. There are three differences, all of them settings in `web_deploy/local/local.env`:

| Setting | Hosted | Dev hardware |
|---|---|---|
| `BOX_MODE` | `vast`: a rented box, woken at sign-in, stopped when idle | `local`: this machine, always on; no idle stop |
| `GATEWAY_HTTPS_ONLY` | `1`: HTTPS behind Caddy, `PUBLIC_URL` is the only accepted origin | `0`: plain HTTP on the LAN address, whatever it is |
| `GPU_SHARING` | `shared`: live models and ingestion each have half of 24 GB | `exclusive`: ingestion takes the whole 12 GB card |

**Why ingestion needs the whole card here.** `llama-server` at the full 128K context with 3 slots takes 8.1 GB. With BGE-M3 and the reranker under load, 0.7 GB of the RTX 3060 is left (measured 2026-10-04). Ingestion's models (Qwen2-VL alone is 7.5 GB) can't fit beside that.

**How the handoff works:**
1. When the ingestion queue gets a batch, the worker asks the API to give up the GPU (`POST /internal/gpu/pause`, service-signed, loopback only).
2. The API stops admitting chat, search and report work, lets work already admitted finish (`LiveGate`), and frees its pinned models.
3. The worker stops `llama-server` and runs the queue. History, evidence, sources and the library stay readable throughout.
4. When the queue is empty, the worker starts `llama-server` again and the API reloads its models (`/resume`).

This happens once per queue, not per batch, because switching costs about a minute. The web app shows a banner while ingestion has the GPU, and a question asked then gets a clear "ingesting" answer instead of an error.

### Setting it up

```bash
docker compose up -d db_dev                                   # firm databases go on your dev Postgres (.env)
venv/bin/pip install -r web_deploy/api/requirements.txt httpx  # the web stack, into the dev venv only
venv/bin/python -m web_deploy.local.run_local setup            # creates local.env, builds llama-server (static, into
                                                               # onyx_data/local-web/llama.cpp) and the web app
venv/bin/python -m web_deploy.local.run_local admin create-owner me@example.com      # + the console's firm password
venv/bin/python -m web_deploy.local.run_local admin create-firm legal-demo "Legal demo" --demo   # + its firm password
venv/bin/python -m web_deploy.local.run_local admin create-user legal-demo me@example.com --role admin
venv/bin/python -m web_deploy.local.run_local                  # Ctrl+C stops everything
```

**Then open it.** Use `http://<this machine's LAN IP>:8000` from any device on your network: the launcher prints the address. Sign in as the firm `legal-demo` with its firm password, then with your email; firm `admin` is the owner's console. Requests from the public pages reach the address in `local.env` once `SMTP_PASSWORD` (a Gmail app password) is filled in; until then they are listed in the console only.

**What runs.** Everything is under `onyx_data/local-web/`: tenants, gateway and queue databases (SQLite), logs, and the `llama-server` build. Three processes run from the appliance venv:
- the API, on `127.0.0.1:8095` only;
- the ingestion worker, which owns `llama-server` here;
- the gateway, on `0.0.0.0:8000` over plain HTTP.

Keep it on a network you trust.

## Running the box

**Startup.** supervisord starts five programs: `postgres`, `llama-server`, `api`, `ingest` and `backup`. The API reports `ready` once the embedder and reranker are pinned and llama-server answers. Signing in wakes the box. The web app shows "Waking the appliance" until the box is ready.

**Idle stop.** The gateway stops the box after `IDLE_MINUTES` with nobody connected, but only when its `/health` says nothing is happening: no question in flight, no report running, and no ingestion running or queued.

**The ingestion queue:**
- One batch at a time, across all firms, first come first served (`onyx_control.ingest_jobs`).
- Each job runs in a fresh child process:
  - with its firm's `POSTGRES_DB` and `ONYX_STORAGE_PATH`, and its batch's language;
  - with the appliance's own in-process backend.
- A box stop interrupts the running job. It is re-queued at the front when the box comes back, and per-source caching skips the work already done.

**Reports.** Each runs in the language it was scoped in, whatever the user picks later. A stopped or interrupted report can be resumed.

## Storage

The vault keeps **one file per recording**, and the upload folder keeps nothing once a batch is ingested. Without this, a video took about 2.4 times its size (measured: 6.04 GB of originals, 14.36 GB in the vault) plus its upload.

| What | Kept? | How |
|---|---|---|
| Upload folder (`uploads/<user>/<batch>`) | No | Ingestion copies each file into the vault. The supervisor deletes the folder once the job is done, failed or cancelled. A job a stop re-queued keeps it, because the retry reads it. An hourly sweep removes folders no queued or running job refers to, untouched for a day (cancelled jobs, uploads never queued). |
| Per-chunk clips | No (`VIDEO_CLIPS_ENABLED=false`) | A citation points at the recording with its start and end; the source page cuts the excerpt (±5 s) when it is opened, cached in the 5 GB source cache. |
| 16 kHz WAV | No (`KEEP_EXTRACTED_AUDIO=false`) | Only the audio stages read it. Deleted once the recording is indexed; a source that failed keeps it for the retry, and a later run re-extracts it if needed. |
| Original video/audio | A compact copy (`COMPACT_MEDIA_ORIGINALS=true`) | Once indexed, replaced by a 720p H.264/AAC MP4 with `+faststart` (audio: AAC m4a, 64 kbit/s), which every browser plays and seeks without a transcode. |
| Frames, documents, images, database | Yes | Small. |

**What the compact copy changes:**
- Ingestion always reads the full-quality upload; the copy is only what is kept and played back. **The full-quality original is not kept on the server.**
- A copy is refused, and the original kept, unless its length is within 0.5 s (or 0.2%) of the original's, because chunk timestamps index into it, and it is at least 20% smaller (`COMPACT_MIN_SAVING`).
- A file a browser already plays at a modest bitrate (an `.mp4` H.264 at most 720p and 3 Mbit/s, or a lean m4a/mp3) is left alone. YouTube downloads usually are.
- The source keeps the hash of its upload, so uploading the same file again still finds it.
- Tuning: `COMPACT_VIDEO_MAX_HEIGHT` (720, the short side, so portrait videos too), `COMPACT_VIDEO_CRF` (26), `COMPACT_AUDIO_BITRATE_KBPS` (64).
- It costs CPU time after indexing (x264 `medium`, niced), not GPU time.

**Material ingested before these settings.** Run the backfill while the ingestion queue is idle:

```bash
set -a; . /workspace/web/api.env; set +a
venv/bin/python -m web_deploy.provision.compact_storage --all --dry-run   # what would be freed
venv/bin/python -m web_deploy.provision.compact_storage --all             # free it (or name firms)
```

Locally, load `web_deploy/local/local.env` instead and add `TENANTS_ROOT=onyx_data/local-web/tenants` and `CONTROL_DATABASE_URL=sqlite:///onyx_data/local-web/control.sqlite3`. Each firm runs in its own child process (`OnyxEngine.compact_storage`); the run is refused while an ingestion is running.

## Backups

Everything a firm has lives on the rented box's disk. A destroyed instance, or one Vast cannot start again, takes it all, so the box and the gateway back up to a **Hetzner Storage Box** with [restic](https://restic.net): encrypted with `RESTIC_PASSWORD` before it leaves the machine, and deduplicated, so after the first run only what changed travels.

| What | From | When | Kept |
|---|---|---|---|
| Every firm database and the queue (`pg_dump`, custom format) | GPU box (`backup` program, `web_deploy/backup/runner.py`) | Every 24 h, **and as soon as an ingestion job has finished**, once the queue is empty | 7 daily, 4 weekly, 6 monthly |
| Every firm's `vault/` and `reports/` | GPU box | Same run | Same |
| The box's `/workspace/web` (`api.env`, TLS key, backup key) | GPU box | Same run | Same |
| The gateway database (firms, accounts, sign-up requests) | VPS (`onyx-web-gateway-backup.timer`, `web_deploy/backup/gateway.py`) | Daily at 03:30 | Same |

Not backed up: `uploads/` (their files are in the vault) and the source cache.

**Keeping the box awake for it.** While a backup is due or running, the API's `/health` says `busy` (`backup_due`, `backup_running`), so the gateway's idle stop waits for new material to be off the box. A failed run is retried after an hour, not at once, so a broken Storage Box can't keep the GPU rented for good; `/workspace/web/backup-status.json` and `/workspace/logs/backup.log` say what happened.

**Setting it up:**
1. Order a Storage Box (BX11, 1 TB, is plenty to start) in the same region as the VPS. Enable SSH support.
2. Fill in `RESTIC_REPOSITORY=sftp:<user>@<user>.your-storagebox.de:onyx-backups` in both `api.env` and `/etc/onyx-web-gateway.env`, and the same `RESTIC_PASSWORD` in both (`web_deploy/provision/prod/` already has one). **Store `RESTIC_PASSWORD` off both machines** (a password manager): without it the backups can't be read.
3. Add the two public keys `provision.sh` and `install_gateway.sh` print (`/workspace/web/backup_ed25519.pub`, `/etc/onyx-web/backup_ed25519.pub`) to the Storage Box: `ssh-copy-id -p 23 -s -i <key> <user>@<user>.your-storagebox.de`, or its `.ssh/authorized_keys`.
4. Check both: `venv/bin/python -m web_deploy.backup.runner --now` on the box (with `api.env` loaded), and `systemctl start onyx-web-gateway-backup` on the VPS. The first run creates the repository.

**Restoring the box** (`web_deploy/backup/restore.py`), onto a new instance:
1. Provision it with the **same** `api.env` (the database password, restic settings and the `WEB_API_SECRET` the gateway knows) and the old backup key (`/workspace/web/backup_ed25519`), or add the new box's key to the Storage Box.
2. `supervisorctl stop api ingest backup` (Postgres keeps running), then, with `api.env` loaded:
   ```bash
   venv/bin/python -m web_deploy.backup.restore --list
   venv/bin/python -m web_deploy.backup.restore --snapshot latest --yes
   supervisorctl start api ingest backup
   ```
   It replaces every firm database and vault with the snapshot's, and puts back the TLS certificate the gateway pins.
3. Point the gateway at the new instance (`VAST_INSTANCE_ID`).

The restore was checked end to end against Postgres 15 with pgvector and restic: back up, drop a table and delete a vault file, restore, and both come back.

## Adding videos from YouTube

On the Ingest page, **Video** batches can also come from YouTube links: single videos, Shorts, youtu.be links and whole playlists. This feature is web only. The on-premise appliance never reaches the internet, and this is the one place the web version does.

- **Checking.** "Check links" resolves the links with yt-dlp without downloading anything (`web_deploy/common/youtube.py`). The user sees every video, can drop some, and is told which links added nothing and why (private, deleted, still live, not YouTube).
  - At most `LINK_MAX_VIDEOS` (200) videos go into one batch.
  - A `watch?v=…&list=…` link adds only that video.
- **Queueing.** The batch folder gets a `.youtube.json` manifest instead of files, and the job is marked `origin: youtube`.
- **Downloading.** When the job reaches the front of the queue, the supervisor first runs `ingest_worker/fetch_links.py` as a child process:
  - It runs **before the GPU is asked for**, so on dev hardware chat keeps answering while the videos download.
  - Videos are fetched one at a time, up to `LINK_MAX_HEIGHT` (720p), as H.264 + AAC MP4.
  - Each file is named `<title> [<video id>].mp4`; the title is what citations show.
  - A video that fails is listed in the batch's report.
  - If none can be fetched, the job fails before the GPU is touched.
  - A stop re-queues the job, and partial downloads resume.
- **Turning it off:** `LINK_INGEST_ENABLED=false` hides the panel and refuses the routes.
- **Keeping it working.** yt-dlp (`yt-dlp[default]`, which includes the challenge solver) and deno (the JavaScript runtime it needs) come from `web_deploy/api/requirements.txt`. YouTube breaks old clients within months: if downloads start failing, run `venv/bin/pip install -U "yt-dlp[default]"` on the box. ffmpeg (already provisioned) merges the video and audio streams.

The Ingest page also takes a **whole folder**, picked or dropped, for every kind of file. Files in subfolders are named after their path inside it (`Unit 3/Part I.mp4` becomes `Unit 3 - Part I.mp4`), so two files with the same name can't overwrite each other.

## Tests

```bash
python3.11 -m venv web_deploy/.venv && web_deploy/.venv/bin/pip install -r web_deploy/requirements-dev.txt
web_deploy/.venv/bin/pytest web_deploy/tests                 # gateway, API (fake engine), queue, worker, signing
cd web_deploy/frontend && npm ci && npm test && npm run typecheck
pytest tests                                                   # the app/ changes (appliance venv)
```

Under the appliance venv, a bare `pytest` skips `web_deploy/tests`, which needs the web stack.

### Local end-to-end without Vast

Run the gateway against a box API on this machine:
- `FAKE_BOX_HOST=127.0.0.1`
- `GATEWAY_HTTPS_ONLY=0`
- no `PUBLIC_URL` and no `BOX_CERT_FILE`

Open `http://localhost:8000` and sign in as the firm, then as yourself.

The box API needs:
- `api.env`'s hosted settings;
- a running `llama-server`;
- uvicorn started without `--ssl-*`.

## Open items

- **Data handling.** The hosted product needs its own data-handling statement. "No data leaves the premises" (CLAUDE.md) is true of the appliance, not of this.
- **Report speed.** Two reports beside live chats share four slots; each slows the others.
