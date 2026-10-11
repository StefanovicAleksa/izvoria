# Hosted Izvoria for many firms (`web_deploy/`)

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
   - A firm's engine is an ordinary `IzvoriaEngine` built by `IzvoriaBootstrapper.build(database=…, vault_dir=…, reports_dir=…, kit=…)`. Every firm shares one `ModelKit`, so a new firm costs a connection pool, not a model load.
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
- **Request Izvoria for your firm** (`/request-access`, `POST /public/signup`). Contact details plus the firm profile from `docs/glossary_authoring/firm_profile_template.md`. The emailed copy follows the template's sections, so it pastes straight into the glossary creator prompt as `{{FIRM_PROFILE}}`. The names section is optional and warns that the form travels by email.
- **Help & support** (`/help`, `POST /public/support`). Linked from both sign-in steps and the request page.

How the public pages are handled:
- **Storage.** Both are stored in the gateway database (`inbound_requests`) and listed under Requests in the owner's console, which shows mail status, a resend, and handled/spam.
- **Email.** They are emailed by `gateway/mailer.py`: plain SMTP, on a background thread, with Reply-To set to the requester.
- **Abuse protection.** Same-origin only, a 32/16 KB body cap, field limits, a hidden honeypot field and a minimum fill time (a bot is told it worked and nothing is kept), 5 per address per hour, and at most 100 emails a day.

**Email settings** (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_STARTTLS`/`SMTP_SSL`, `MAIL_FROM`, `SIGNUP_TO`, `SUPPORT_TO`):
- Mail is off when there is no `SMTP_HOST`, or when there is a user without a password: requests are then only stored.
- Hetzner Cloud blocks outbound 25 and 465 on new projects, so use 587 with STARTTLS.
- Gmail (dev) needs 2-step verification and an app password, and `MAIL_FROM` must be the same address.

**Look.** The web app copies the desktop app (`frontend/izvoria_desktop/ui/theme.py`):
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

It needs **only a rented GPU box**: no Hetzner VPS, no domain, no accounts. The test signs its own requests to the box's API with `WEB_API_SECRET`, exactly as the gateway would. It is one command, `web_deploy/spike` (`python -m web_deploy.spike`), in three stages:

| Stage | What it answers | Needs |
|---|---|---|
| `sweep` | llama-server's VRAM at each slot count and shared KV size (default grid: 1–6 slots, 128K–256K tokens); whether `--kv-unified` lets one slot use the whole cache; **which `LLM_SERVER_PARALLEL` / `LLM_SERVER_KV_TOKENS` to use** (most slots, then the biggest cache, that keeps 128K per request and the live side under 13.5 GB) | the 9B GGUF |
| `stress` | at that size: the JSON grammar through the app's own client; throughput one after another vs at once; a ~120K-token request beside three short chats; more long requests than the cache holds, sent straight to the server and again through `KvTokenBudget` | the 9B GGUF |
| `shared` | **the production layout itself**: the box running as provisioned, a facts document ingested, then a video ingested on the ingestion half while three users keep chatting on the live half; VRAM peak, failed chats, whether the answers stay right | every model |

`all` runs the three in order: it stops the supervisord programs for the first two, writes the sweep's pick into `api.env` (the original is kept as `api.env.before-spike`), and starts them again for the third. The result is `/workspace/spike/report.md` (verdicts first, then the numbers) and `results.json`.

**Steps:**
1. Rent the box on Vast: one RTX 3090, the `*-cudnn-devel-ubuntu22.04` image, about 120 GB of disk, port 8095 exposed. Prefer a host with fast download and upload: the models travel from your machine.
2. Copy the code and models, then provision (as in "Setting it up"; `provision.sh` is happy with the example settings, or with `prod/api.env` copied in first):
   ```bash
   web_deploy/provision/sync_to_box.sh root@<ip> <port>            # all models (51 GB) for the full test
   web_deploy/provision/sync_to_box.sh root@<ip> <port> --live     # or just the live models (12 GB): sweep + stress only
   ssh -p <port> root@<ip> 'bash /workspace/onyx_v3/web_deploy/provision/provision.sh && bash /workspace/onyx_v3/web_deploy/provision/onstart.sh'
   ```
3. Put a real recording on the box for the shared test (5–20 minutes of speech with slides is a fair test; without one it uses the 15-second test clip):
   ```bash
   scp -P <port> lecture.mp4 root@<ip>:/workspace/samples/
   ```
4. Run it (about 1–1.5 h for `all`; `tmux` keeps it going if the SSH session drops):
   ```bash
   cd /workspace/onyx_v3 && set -a && . /workspace/web/api.env && set +a
   venv/bin/python -m web_deploy.spike all --video /workspace/samples/lecture.mp4
   ```
5. Copy the report home (`scp -P <port> root@<ip>:/workspace/spike/report.md .`), record the numbers here, put the chosen `LLM_SERVER_PARALLEL` / `LLM_SERVER_KV_TOKENS` into `prod/api.env`, then **destroy the instance** (a stopped one still bills its disk).

**Cost:** a 3090 rents for roughly $0.15–0.40 an hour, so the whole test is a few dollars; the slowest part is usually uploading the models.

**Reading it.** Every verdict line should say PASS. The ones that need a decision:
- *No layout with 128K per request fits:* the fallback (`--parallel 2` without `--kv-unified`) caps a request's context below 128K, and that needs a decision.
- *A video doesn't ingest beside chat, or chats fail during it:* the 24 GB split doesn't hold. Lower `LLM_SERVER_KV_TOKENS` (not a request's 128K) and re-run `shared`, or set `GPU_SHARING=exclusive` (ingestion pauses chat, as on the dev card).
- *The JSON grammar fails:* the pinned llama.cpp release behaves differently on this machine; don't move the pin without re-running this.

The retrieval scrubber's quality on the 9B (`scripts/eval/rag_eval.py`) is not part of it: it needs the labelled dev corpus, and the dev-card run below already compared its verdicts.

**A trial on the 12 GB dev card (2026-10-09)** ran all three stages at smaller sizes (`sweep --grid 2:65536,4:65536,4:98304`, `stress --kv 65536 --long-tokens 40000`, `shared` against `run_local` with `--api-url http://127.0.0.1:8095`). It found:
- **llama-server rejects an over-full cache; it doesn't queue.** Two 40K-token requests sent at once into a 64K cache both failed ("Context size has been exceeded"). Through the app's client, `KvTokenBudget` made the second wait, and both were answered. So the budget is load-bearing: never call llama-server around it.
  - Checked around it, all through the app's client in a 64K cache: two 52K requests one after the other, and one 28K request followed by two more at once, all answered. The server frees an idle slot's cached prompt when a new request needs the room, so `cache_prompt` doesn't defeat the budget. A 64K prompt allowed a 3K reply ends that reply where the cache is full (`finish_reason: length`), not with an error.
  - `test_a_request_the_shared_cache_cannot_hold_yet_waits_instead_of_reaching_the_server` fails if a request ever goes around the budget.
- Short chats beside a 40K-token request waited about 37 s. Fixed by the scheduler patch below.
- Four answers ran 48 tokens/s one after another and 113 tokens/s at once.

### The 3090 run (2026-10-10): passed

`python -m web_deploy.spike all` on a rented RTX 3090 (Vast, Hungary, driver CUDA 12.7, Ubuntu 24.04), with a 16-minute lecture (speech + slides). Every verdict passed.

| | Measured |
|---|---|
| llama-server VRAM | 8.1 GB (1–4 slots, 128K) … 9.8 GB (6 slots, 192K) … 11.2 GB (6 slots, 256K) |
| **Chosen layout** | **`LLM_SERVER_PARALLEL=6`, `LLM_SERVER_KV_TOKENS=196608`**: live side ~12.8 GB, every request keeps 128K |
| Throughput | 97 tok/s one at a time, 213 tok/s four at once |
| A 120K-token request beside three chats | all answered; the chats in 3.6 s |
| Chats beside a long request, scheduler patch off → on | 61.6 s → 6.5 s; the long request 62.1 → 62.3 s |
| **Ingestion beside live chat** | the lecture indexed in 4.1 min while 3 users chatted: **147 of 147 chats answered** (median 1.6 s, slowest 4.8 s), 49 of 49 correct; **peak VRAM 20.2 GB of 24** (Qwen2-VL 7.9 GB beside the live side's 12.3 GB) |

Four things a fresh box got wrong, now fixed (each would break every new deploy):
- **SQLAlchemy 2.1** (unpinned) made psycopg 3 the default driver for `postgresql://`: every database URL now names `postgresql+psycopg2`.
- **dpkg's config question** stopped `provision.sh` on Vast's base image, which ships its own `supervisord.conf`: apt now keeps existing config files.
- **PyPI's PyTorch targets CUDA 13**, and the host's driver stopped at 12.7, so the API refused to start; torchcodec's CUDA-13 build couldn't even load. `provision.sh` reinstalls torch, torchaudio, torchvision and torchcodec as the same versions' CUDA 12.6 builds and checks the GPU. Rent hosts whose "Max CUDA" is 12.6 or more.
- The report counted a finished ingestion job that indexed nothing as a pass; it now needs an indexed file. Not a deploy bug, but it hid the torchcodec failure for one run.

Not measured here: the over-full cache. Its two long requests shared one cached prompt, so the cache never filled; the stress stage now sends different prompts. The dev card's run below shows what overflowing does, and `KvTokenBudget`'s test guards it.

**Getting models onto a rented box.** Uploading 51 GB from a home connection (~2 MB/s up) takes ~7 hours. Instead, the box downloaded the 12 models that exist on Hugging Face (37 GB, every file checked against the local copy's hash) in 8 minutes; only the locally quantized Qwen2-VL (6.3 GB) and the small NeMo/OCR files were uploaded. Gemma 4 E4B is only needed for Serbian.

### llama-server patch: answers before long prompts

llama-server is built with Izvoria's own patch, `web_deploy/provision/llama_patches/0001-server-answers-before-long-prompts.patch`, which `build_llama_server.sh` applies to the pinned release.

**The problem.** In every step llama-server first adds one token for each answer being written, then fills the batch with prompts still being read. A step finishes only when everything in it is done. While someone's long prompt (a report section, a long conversation) is being read, each answer token waits for a 2K-token chunk of it: answers dropped from ~50 to ~5 tokens a second, and chats asked beside a 40K-token request took 43 s instead of 5.

**The patch** (only `update_slots()`'s prompt loop in `tools/server/server-context.cpp`):
- Prompts still to be read are taken shortest-remaining-first (stable), instead of in slot order.
- While any answer is being written, a prompt with more than `ONYX_SHORT_PROMPT_TOKENS` left to read waits, and reads at most `ONYX_LONG_PROMPT_TRICKLE` tokens a step. With no trickle it skips the step before anything else runs for it: no state change, no checkpoint. Shorter prompts are still read in full batches, so a new chat starts at once. Once no answer is being written, long prompts read in full batches again.
- Both variables unset (or 0): llama-server's own behaviour, apart from the order.

**Measured on the dev card** (40K-token request, then three chats over ~4K tokens of evidence 2 s later):

| | Chats | The long request |
|---|---|---|
| unpatched | 43 s | 41 s |
| patched, unset | 42 s | 40 s |
| `ONYX_SHORT_PROMPT_TOKENS=8192`, no trickle | **13 s** | 44.5 s |
| same, 64-token trickle | 21 s | 46.5 s |

So the settings are `ONYX_SHORT_PROMPT_TOKENS=16384` and `ONYX_LONG_PROMPT_TRICKLE=0`, in `api.env` / `local.env`. A long request is held back only while answers are being written: in a firm that chats constantly, a report section waits its turn.

**Checked:** llama.cpp's own server tests (`tools/server/tests`: completion, chat, streaming, context shift, slot save, KV, infill, tokenize, embeddings, rerank, speculative) give the same result on the patched build, unset and with the rule forced on every prompt, as on the unpatched one. The spike's stress stage re-measures the rule on the 3090.

**Moving the pin** means re-applying the patch (`build_llama_server.sh` refuses one that doesn't apply) and re-running the spike's stress stage.

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

Locally, load `web_deploy/local/local.env` instead and add `TENANTS_ROOT=onyx_data/local-web/tenants` and `CONTROL_DATABASE_URL=sqlite:///onyx_data/local-web/control.sqlite3`. Each firm runs in its own child process (`IzvoriaEngine.compact_storage`); the run is refused while an ingestion is running.

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
