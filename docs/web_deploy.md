# Hosted Izvoria for many firms (`web_deploy/`)

This is the hosted web version that demos are shown on and that users use. It runs on two machines:

- **A Hetzner VPS.** Always on. It handles firms, users and sign-in, wakes the GPU box, and **holds every firm's data**: the databases, the vaults, the uploads and the backups.
- **A rented Vast.ai GPU box.** On demand. It runs the models and the appliance's code, and **keeps no client data at rest** (see "Where data lives").

Everyone signs in at one address. Each firm has its own sign-in name and shared password, its own database and its own file vault, shared by all of that firm's users. Several users can chat, search and run reports at the same time. Ingestion waits its turn in a queue.

It sits beside the single-tester noVNC demo (`deploy/`, `docs/deploy.md`) and replaces it. It also reuses four of the demo's modules:

- the Vast client;
- the source vault check;
- the ffmpeg media preparer;
- upload filename sanitising.

It is ops tooling, not the product's security model. The appliance proper stays on-premise and air-gapped, and nothing in `web_deploy/` is installed there.

## How a question travels

```
browser ─https─▶ Caddy (VPS) ─▶ gateway (VPS) ──────────────▶ 127.0.0.1:18095 ══ ssh -R ══▶ API (GPU box, loopback)
                 one host, TLS   firms, users, sessions (SQLite)                               one process, all firms
                                 uploads ─▶ /workspace/tenants (vaults, uploads)               ├─ firm engine …
                                 cited sources, cut from the vault here                        └─ shared model kit
                                 Postgres 15 + pgvector (every firm) ◀══ ssh -L 15432 ══ box      ├─ BGE-M3 + reranker
                                 backups ─▶ Storage Box                                           └─ llama-server
                                 Vast start/stop, idle stop          ◀══ rsync (one job's files) ══ ingestion worker
```

1. **The gateway identifies the firm and the user.**
   - The session names the user, and the user's account names the firm (see "Signing in" below).
   - The firm and the account are re-read on every request, so disabling either takes effect at once.
   - The gateway forwards `/api/*` to the box with an HMAC over method, path, query, firm, user, role and timestamp (`web_deploy/common/signing.py`). It travels through the box's own SSH tunnel; the box has no public port.
   - The box never sees a cookie or a password.
   - Uploads and cited files are the exception: the gateway handles them itself, because the files live on the VPS (`web_deploy/gateway/data.py`).
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
   - A cited source is re-resolved by chunk id on the box (`/api/sources/{chunk}/locate`: the file's path and the cited span), then cut and served by the gateway from the VPS's vault. The path must lie inside the firm's own vault there, so not even the box can make the gateway serve another firm's file.

## Where data lives

The data used to live on the rented box's disk. We can't vouch for what a Vast host does with a disk, so the box now only computes.

| What | Where | Why there |
|---|---|---|
| Firm databases (`onyx_firm_<slug>`), the ingestion queue (`onyx_control`) | Postgres 15 on the VPS, loopback only | The box reaches it through its tunnel |
| Vaults, uploads | `/workspace/tenants` on the VPS (a Hetzner Volume) | The gateway receives uploads and serves cited sources from here |
| Source cache (cuts of cited recordings) | `/workspace/source-cache` on the VPS | Cut by the gateway's ffmpeg |
| Backups | Hetzner Storage Box, from the VPS | Everything to back up is on the VPS |
| Report Markdown copies | Nowhere (`REPORT_FILES_ENABLED=false` on the box) | Downloads render from the database row |
| Models, code, settings, keys | The box's `/workspace` | Not client data |

**The link: two SSH sessions the box opens**, so file transfers never delay chat:
- `tunnel` (`box_tunnel.sh`): `-L 127.0.0.1:15432` reaches the VPS's Postgres, `-L 127.0.0.1:18000` the gateway (the box's database password, its requests for file transfers), and `-R 127.0.0.1:18095` lets the gateway reach the box's API.
- `tunnel-files` (`box_tunnel_files.sh`): `-R 127.0.0.1:18022` lets the VPS reach the box's own sshd, through which it moves one job's files.
- SSH, not WireGuard: a Vast box is an unprivileged container, with no `/dev/net/tun` or `NET_ADMIN`, and often no FUSE. A plain outbound `ssh` needs none of them, and it lets the box expose no port at all.
- The box's user on the VPS (`izvoria-box`) has **one key, for those ports only**: no shell, no files (`/etc/onyx-web/box_authorized_keys`, `add_box_keys.sh`).
- **Everything works only while the gateway has the box started.** The gateway writes `open` to `/var/lib/onyx-web-box/access` before it starts the box, and `closed` once it has stopped it, or finds it truly stopped (`exited`/`stopped`/`offline`, not a reload; `gateway/box.py`). Root applies every change (`izvoria-box-access`, run by `izvoria-box-access.path`):
  - **open:** the box's database role `izvoria_box` may log in, with a **new random password**; sshd offers the box's key (`izvoria-box-keys`).
  - **closed:** the role may not log in and has no password; **its database sessions and the box's SSH sessions are cut**; sshd no longer offers the key.

  A stopped box, or anything copied off its disk, reaches nothing.
- **No database password on the box**, on disk or in any process environment. Each box process asks the gateway for this start's password whenever it opens a new database connection (`web_deploy/common/box_credentials.py` → `app/core/database/data/credentials.py`, a `do_connect` hook on every engine). Pools reuse connections, so this is a local call now and then. The gateway answers only a signed, loopback caller while the box is started (`gateway/box_internal.py`; Caddy refuses `/internal/*` from outside).
- **No key to the VPS's files on the box.** The VPS moves each job's files itself (below).
- The VPS's host key is pinned on the box (`/workspace/web/known_hosts`, from `provision.sh`; compare its fingerprint with the VPS's).

**Ingestion is staged, by the VPS** (`gateway/staging_server.py`; the box's side is `ingest_worker/staging.py`). Around each job the box asks, and the VPS decides what moves, only for the job that is running:
1. **In:** the VPS sends the batch's uploads into the box's scratch, and the vault files of any source with the same content (its original, frames, a kept WAV, clips), found by hash in the firm's database, so a retry or a re-upload finds its files where the database says they are. Nothing of another firm, or another job, travels.
2. The ingestion child runs as before. Its rows go straight into the VPS database, its files into scratch.
3. **Out:** the VPS takes the firm's scratch vault back, deletes on its side the files it sent that the box no longer has (a compaction swap of a pulled original), and the box wipes its scratch.

Before every transfer the ingest worker lets the VPS's staging key in for root on the box (`ensure_access`), so the VPS can reach the scratch through `tunnel-files`. Not just at start: Vast rewrites `authorized_keys` when it restarts the container, and on 2026-10-20 that dropped the key under a running worker, so every transfer was refused with `Permission denied (publickey)`.

**Stopping a batch and deleting a file.** A running batch can be stopped by whoever queued it or a firm admin (the Ingest page's Stop: `request_stop` on the job row). The supervisor checks every 3 s, terminates the child (killed after 30 s), removes the file that was half-way through (sources created during the run with no chunks, `web_deploy/common/source_removal.py`), marks the job cancelled and stages out as usual: files the batch finished stay. Firm admins delete a file from the Library on the **VPS** (`DELETE /api/library/{source}` in `gateway/data.py`), not on the box: the rows go (the `sources` cascade, plus `jobs` and the extracted `files` rows it misses), then its vault files. Refused while the firm has a batch running. Reports keep their denormalised citations; a chat citation to a deleted file no longer opens.

A job whose files never arrived is dropped from scratch at once (`Stager.abandon`): nothing ran, so nothing goes back. Any other job's results stay pending until the VPS has them, and the worker takes no new job meanwhile. That protects files that exist only on the box, but it also holds up every firm's batches, so a push that keeps failing needs fixing on the spot (`/workspace/logs/ingest.log`).

### Videos are prepared on the VPS (`PREPARE_VIDEOS`)

Raw footage can run to ~100 GB an hour, and the box needs none of that resolution: the vault keeps a 720p copy anyway. With `PREPARE_VIDEOS=true` the gateway makes that copy **before** the box sees the recording (`web_deploy/gateway/prepare.py`), so only the copy is staged to the box, ingested and kept. Transfer to the rented GPU, which its host bills per byte, shrinks accordingly.

- A video batch is queued as `preparing`. The box claims only `queued` jobs, and isn't woken until the batch is ready. The Ingest page shows "Making the 720p copies · 2/5", and the job can be cancelled meanwhile. The wait list and cancelling are answered by the gateway, so watching it doesn't wake the box.
- Per video: ffprobe; an MP4 that is already compact (the box's own rule, `is_compact_video` in `app/features/media_compaction/utils/ffmpeg_args.py`) is left alone; otherwise the original is hashed, encoded with the box compaction's own arguments (`compact_video_args`, at `PREPARE_PRESET` and 128 kbit/s audio for Whisper), checked for length (±0.5 s / 0.2 %), and swapped in. `talk.mov` becomes `talk.mp4`, or `talk-2.mp4` if the batch already has one.
- **Duplicate detection still works.** A source is matched by the hash of what was *uploaded*, and ffmpeg's output isn't reproducible. So the preparer leaves an upload record beside the copies (`.upload-record.json`, `app/features/storage/utils/upload_record.py`): each copy's original hash and name. `StorageAPI.ingest_local_file` uses it instead of hashing the copy, and the source keeps the uploaded name. Staging looks up a batch's earlier sources by the same hashes.
- **The box re-encodes nothing.** Its compaction sees an already-compact MP4 and keeps it. The box ingests exactly the file that is kept, so the timeline check compaction exists for can't fail.
- **If the copy fails** (ffmpeg error, a length that drifted), the original stays and the box ingests and compacts it as before.
- **After a gateway restart**, a partial copy is removed, a recorded copy is kept (its original deleted if it was still there), and the rest is done again.
- **Only video.** Audio uploads are small, and re-encoding them before transcription could cost accuracy. Documents, images and YouTube batches (already 720p) pass straight through.
- **Cost:** the VPS's CPU. x264 runs niced with every core: a dedicated server with many cores prepares a long batch far faster than a small cloud VPS. The full-size original sits on the VPS only until its copy is made.
- **Resolution costs the vision model nothing.** Qwen2-VL never sees more than about 720p anyway: a lone frame is capped at 1280×28×28 ≈ 1.0 MP (`_MAX_PIXELS` in `vlm_adapter.py`, ~1334×750 at 16:9, above 720p's 0.92 MP), and frames sent together at `VISION_MULTI_IMAGE_MAX_PIXELS` ≈ 0.6 MP (~1035×582, below 720p). Video frames get no OCR. What does change is that frames come from an H.264 encode (CRF 26, `PREPARE_PRESET`) instead of the upload, so very small slide text can pick up compression artifacts. A quick comparison on one dense-slide lecture is enough before turning it on; lower `PREPARE_VIDEO_CRF` if text smears.

The scratch sits at **the same absolute path on both machines** (`/workspace/tenants`): the database stores absolute file paths, so a path written on the box is valid on the VPS once pushed, and nothing is ever rewritten.

A marker per job (`/workspace/staging-state`) lives until the VPS has taken the results back. A crash, a host stop or a failed push leaves the job recoverable: the worker pushes and wipes before it takes new work, at the next start too, and the API's `/health` says `staging_pending` meanwhile, which keeps the gateway's idle stop away. Until a push lands, a citation to one of its new files answers "not available yet" rather than failing.

**What this does and doesn't protect.** The agreed bar is *nothing persisted on the rented box, and everything in transit encrypted*. Two exposures remain, and the data-handling statement must say so:
1. **Memory while working.** Audio, document text, questions, retrieved excerpts and answers are in plain form in the box's RAM and VRAM while they are processed. A host that inspects a running container's memory could read them. Only confidential-computing GPUs close this, not consumer cards.
2. **The database while the box runs.** The box's processes hold this start's database password in memory. Someone with root on the host who deliberately dumps a process's memory could take it and read every firm's database **until the box is stopped** (then the role is disabled, the password changed and every session cut). There is no file to copy and no environment variable to read; it takes a deliberate attack on a running process. Only moving the engine itself off the box (the box as a pure model server) would close this.

Also worth knowing:
- The scratch is on the box's disk, because a symlink to `/dev/shm` would change the stored paths (the vault resolves its root). It is wiped after every job, but deleting from an SSD is not a secure erase.
- `/workspace/tmp` (`TMPDIR`) is emptied at every fresh start.
- Logs (`/workspace/logs`) rotate at 20 MB.

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

## Firm glossaries

Each firm can have its own glossary per served language: the names and terms its recordings are transcribed (Whisper hotwords) and corrected (domain phonetic correction) with. Without one, a firm gets the shared `data/glossaries/<lang>/general.json`, as every firm did before; Serbian has no general glossary, so it gets none. Documents and images don't use a glossary.

**Setting one: the owner only.**
- **Owner console:** Firms → *Glossary* on the firm's row. Per language: upload a JSON file, pick a built-in (`finance`, `medical`, `prop_trading_macro`, …: whatever is in `data/glossaries/<lang>/`), or go back to *General (shared)*. A firm's own administrators can't see or change it (`/admin/api/firms/{slug}/glossaries[/{lang}]`, owner-only).
- **`admin_cli`** on the VPS, as the gateway's user (the files share the box staging's group): `sudo -u onyx /opt/onyx-web-gateway/bin/python -m web_deploy.gateway.admin_cli glossaries <slug>`, `set-glossary <slug> <lang> file.json`, `set-glossary <slug> <lang> --builtin medical`, `set-glossary <slug> --from-claude answer.json` (every language of a glossary creator prompt answer at once, all or nothing), `clear-glossary <slug> <lang>`.

**Checks.** An upload goes through `scripts/validate_glossary.py`: any error refuses it and is listed (UPPER_SNAKE_CASE labels, duplicates, Cyrillic in `sr`, …), warnings are kept and shown. It needs terms, at most 2 MB, and an industry name (its `"industry"` key, else the file's name), which becomes the file name. Built-ins are taken as shipped.

**Where it lives and how it reaches a job** (`web_deploy/gateway/glossaries.py`):
- `/workspace/tenants/<firm>/glossaries/<lang>/<industry>.json` plus `meta.json` (source, counts, warnings). At most one per language; a new one replaces the folder atomically.
- The VPS staging sends the job's language's folder with the job (`staging_server.stage_in`). The box drops any glossary of the firm before asking, and wipes it with the rest of the scratch afterwards: it holds client names.
- The ingestion child gets `GLOSSARY_PATH=<tenants>/<firm>/glossaries` and `APPLIANCE_INDUSTRY=<INDUSTRY>` (`supervisor.firm_environment`), which is exactly how the appliance reads its own `data/glossaries/`. No `app/` change.
- **It applies to recordings ingested from now on.** Domain adaptation is cached per source, so a recording already indexed keeps the glossary it was read with until it is re-ingested.

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
| `REPORT_FILES_ENABLED` | true | false | No Markdown copy of a report on the box's disk (`NoReportFileStore`); the web app renders downloads from the row. |

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

It was run before the data moved to the VPS, and needed **only a rented GPU box**: no Hetzner VPS, no domain, no accounts. `sweep` and `stress` still need nothing else. `shared` seeds files into the box's own tenants and expects a database beside them, so it is not yet adapted to the new layout. The test signs its own requests to the box's API with `WEB_API_SECRET`, exactly as the gateway would. It is one command, `web_deploy/spike` (`python -m web_deploy.spike`), in three stages:

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

**Getting models onto a rented box.** Uploading 51 GB from a home connection (~2 MB/s up) takes ~7 hours. Instead, the box downloaded the 12 models that exist on Hugging Face (37 GB, every file checked against the local copy's hash) in 8 minutes; only the locally quantized Qwen2-VL (6.3 GB) and the small NeMo/OCR files were uploaded. Gemma 4 E4B is only needed for Serbian. This is now `fetch_models.py` plus `sync_to_box.sh --local-only` (see "Setting it up").

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

**Settings made ahead of time (optional).** `web_deploy/provision/prod/` (gitignored, never synced) can hold the two production files, made from the examples:
- `gateway.env` → `/etc/onyx-web-gateway.env` on the VPS (`chmod 600`), copied **before** `install_gateway.sh` (which then keeps it instead of generating secrets);
- `api.env` → `/workspace/web/api.env` on the GPU box, copied **before** `provision.sh`, with the same `WEB_API_SECRET` and the VPS's address in `VPS_HOST` (and **no** database password).

Never copy `prod/` to either machine as part of the repo: the gateway file holds the Vast API key.

### Gateway and data (Hetzner VPS), first

1. Attach a Hetzner Volume and mount it at `/workspace` (vaults and uploads grow with every firm). The VPS also runs Postgres, pgvector search and the ffmpeg cuts now: 4 vCPU and 8–16 GB RAM is a sensible start.
2. DNS: one A record for `onyx.example.com`, pointing at the VPS.
3. Copy the repo, with the built `web_deploy/gateway/static`, to `/opt/onyx_v3`. Then:
   ```bash
   sudo bash /opt/onyx_v3/web_deploy/provision/install_gateway.sh onyx.example.com
   ```
   This installs Postgres 15 with pgvector (loopback only; the role, `onyx_control`, `vector` in `template1`), the data directories (group `izvoria-data`), the `izvoria-box` account with its sshd rules and `izvoria-box-keys`, the gateway, the backup service and Caddy. It writes `/etc/onyx-web-gateway.env` with fresh `WEB_API_SECRET`, `POSTGRES_PASSWORD` and `RESTIC_PASSWORD`, and prints the SSH host key's fingerprint.
4. Fill in `/etc/onyx-web-gateway.env`: `VAST_API_KEY`, `VAST_INSTANCE_ID`, and `SMTP_*`, `MAIL_FROM`, `SIGNUP_TO`, `SUPPORT_TO` for the request and help pages. Then `systemctl restart onyx-web-gateway`.
5. Create your owner account (it also asks for the console's firm password):
   ```bash
   cd /opt/onyx_v3 && sudo -u onyx GATEWAY_DB=/var/lib/onyx-web/gateway.sqlite3 \
       /opt/onyx-web-gateway/bin/python -m web_deploy.gateway.admin_cli create-owner you@example.com
   ```
6. Sign in at `https://onyx.example.com` as the firm `admin`, then with your email. Create firms, each with a firm password, and their first administrators. `admin_cli` does the same from the shell (`create-firm … --generate` makes up the firm password and prints it once).

Caddy gets and renews the one certificate for that host automatically.

### GPU box (Vast.ai)

Template:
- an nvidia/cuda `*-cudnn-devel-ubuntu22.04` image;
- one 24 GB GPU, on a host in the EU near the VPS's Hetzner location: every database query now crosses the tunnel;
- **no ports exposed** (the box only dials out; Vast's own SSH stays for you);
- on-start script `bash /workspace/onyx_v3/web_deploy/provision/onstart.sh`.

Steps:
1. Build the web app on the dev box:
   ```bash
   cd web_deploy/frontend && npm ci && npm run build   # -> web_deploy/gateway/static
   ```
2. Copy the code, and the ~6 GB of models that exist only on the dev box (the 4-bit Qwen2-VL, the NeMo VAD and TitaNet files, RapidOCR):
   ```bash
   web_deploy/provision/sync_to_box.sh root@<ip> <ssh port> --local-only
   ```
   Meanwhile, on the box, fetch the other ~49 GB from Hugging Face (minutes on a data-centre line; it needs only the system Python, so it can run beside step 3):
   ```bash
   python3 /workspace/onyx_v3/web_deploy/provision/fetch_models.py
   python3 /workspace/onyx_v3/web_deploy/provision/fetch_models.py --verify   # once both are done
   ```
   Each model is pinned to the exact commit the dev box downloaded, and every file is checked against `web_deploy/provision/models.lock.json` (size and sha256 of the dev box's copies), so the box runs byte-identical models. A download that stops resumes. After changing a model on the dev box, update `HUGGING_FACE` in `fetch_models.py` and run `venv/bin/python -m web_deploy.provision.fetch_models --write-lock`. (Without `--local-only`, `sync_to_box.sh` still uploads all 51 GB from the dev box.)
3. Provision the box, naming the VPS:
   ```bash
   ssh … 'VPS_HOST=onyx.example.com bash /workspace/onyx_v3/web_deploy/provision/provision.sh'
   ```
   This installs the appliance venv, builds `llama-server`, writes `/workspace/web/api.env`, makes the box's tunnel key, and pins the VPS's host key. **Compare the fingerprint it prints with the one `install_gateway.sh` printed.**
4. Do what it prints:
   - on the VPS, as root: `izvoria-add-box-keys '<tunnel key>'`;
   - into the box's `api.env`: `WEB_API_SECRET` from `/etc/onyx-web-gateway.env` (no database password: it gets one at every start).
5. Start it: `bash /workspace/onyx_v3/web_deploy/provision/onstart.sh`, or let the gateway start it at the next sign-in. The tunnel comes up only while the gateway has the box started (see "Where data lives").

A replacement box needs only steps 2–4: it holds nothing to restore.

**Spare boxes.** A stopped Vast instance keeps its disk but not its GPU: anyone may rent that GPU meanwhile, and the instance can't start until they release it (on 2026-10-14 the primary's was taken for the rest of the day). So keep one or two spares: fully provisioned boxes on **other machines**, each costing only its disk while stopped (~$0.03/hr). Provision each like the primary (steps 2–4, its own tunnel key added with `izvoria-add-box-keys`), then list them all:
```
VAST_INSTANCE_ID=55919995,55823057,56923441
```
The gateway (`gateway/vast_pool.py`) then:
- **wakes the cheapest box whose GPU is free.** At each wake it ranks the boxes by the hourly price Vast quotes for each right then (`dph_total`; hosts change prices), cheapest first; the list's order breaks ties and ranks a box with no quoted price last. A running box is never swapped for a cheaper one. A box Vast refuses, or leaves "scheduling" past `scheduling_grace_seconds` (60 s), is stopped again at once (Vast queues a refused start and would run it later) and the next is tried. With every GPU taken, the web app says so on its "asleep" card;
- **keeps at most one running.** Any other box found running, or with a start pending, is stopped, so a second GPU is never billed;
- **stops them all** at the idle stop.
Every box holds the same code and models and no data, so which one answers makes no difference. One listing call (`GET /api/v1/instances/`) reads every box's state, so polling stays within Vast's rate limit. Each box is still capped by its host's maximum duration (shown when renting): replace one before its date.

### Moving existing data

A box provisioned before this change kept its own Postgres (`/workspace/pgdata`) and vault. To move a firm's data:
1. On the old box, take a backup (`venv/bin/python -m web_deploy.backup.runner --now`, with its old `api.env`).
2. On the VPS, restore it with `BACKUP_HOST_NAME=onyx-box` (see "Backups"). The paths stay valid, because `/workspace/tenants` is the same path on both machines.
3. Give the restored files to the shared group: `chgrp -R izvoria-data /workspace/tenants && chmod -R g+rwX /workspace/tenants`.
4. Destroy the old instance. A stopped one keeps its disk.

### Upgrading a deployment from subdomains

Firms used to live at `<firm>.<domain>`.
1. Deploy the code. The gateway database gains its new columns at startup, and existing firms have no firm password, so nobody can sign in yet.
2. Run `admin_cli set-firm-password <slug>` for every firm (`list` flags the ones missing), and `set-firm-password admin` for the console.
3. Install the new Caddyfile (`install_gateway.sh` does it) and replace `BASE_DOMAIN` with `PUBLIC_URL` (an old `BASE_DOMAIN` still works, as `https://<BASE_DOMAIN>`). The wildcard DNS record can go.
4. Tell each firm its sign-in name and password. Everyone signs in again, because the old cookies belonged to the subdomains.

### Demo firms

Create the firm on the gateway (`legal-demo`, "Demo"), then queue sample material **on the VPS**, where uploads and the queue live:
```bash
cd /opt/onyx_v3 && set -a && . /etc/onyx-web-gateway.env && set +a
sudo -u onyx /opt/onyx-web-gateway/bin/python -m web_deploy.provision.seed_demo legal-demo document en /srv/samples/contracts
sudo -u onyx /opt/onyx-web-gateway/bin/python -m web_deploy.provision.seed_demo clinic-demo audio sr /srv/samples/consultations
```
The box ingests them when it next runs.

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

**What runs.** Everything is under `onyx_data/local-web/`: tenants, gateway and queue databases (SQLite), logs, and the `llama-server` build. The gateway receives uploads and serves cited sources here too, exactly as on the VPS, and staging is off (`STAGING_REMOTE` unset): the box and the data share one disk. Three processes run from the appliance venv:
- the API, on `127.0.0.1:8095` only;
- the ingestion worker, which owns `llama-server` here;
- the gateway, on `0.0.0.0:8000` over plain HTTP.

Keep it on a network you trust.

## Running the box

**Startup.** supervisord starts four programs: `tunnel`, `llama-server`, `api` and `ingest`. The API and the worker wait for the tunnel's end of Postgres (`wait_for_tunnel.sh`). The API reports `ready` once the embedder and reranker are pinned and llama-server answers. Signing in wakes the box. The web app shows "Waking the appliance" until the box is ready.

**What works while the box sleeps.** Uploading and queueing (the gateway writes the files and the queue row on the VPS, then wakes the box). Everything else needs the box: chat, search, reports, history and opening a cited source (the box says which file a chunk cites).

**Idle stop.** The gateway stops the box after `IDLE_MINUTES` with nobody connected, but only when its `/health` says nothing is happening: no question in flight, no report running, no ingestion running or queued, and no results waiting to be pushed to the VPS (`staging_pending`).

**The ingestion queue:**
- One batch at a time, across all firms, first come first served (`onyx_control.ingest_jobs`).
- Each job runs in a fresh child process:
  - with its firm's `POSTGRES_DB` and `ONYX_STORAGE_PATH`, and its batch's language;
  - with the appliance's own in-process backend;
  - between the staging in and out described in "Where data lives".
- A box stop interrupts the running job. What it had written is pushed to the VPS at the next start, the job is re-queued at the front, and per-source caching skips the work already done.

**Reports.** Each runs in the language it was scoped in, whatever the user picks later. A stopped or interrupted report can be resumed.

## Storage

The vault keeps **one file per recording**, and the upload folder keeps nothing once a batch is ingested. Without this, a video took about 2.4 times its size (measured: 6.04 GB of originals, 14.36 GB in the vault) plus its upload.

| What | Kept? | How |
|---|---|---|
| Upload folder (`uploads/<user>/<batch>`, on the VPS) | No | Ingestion copies each file into the vault. The gateway's janitor (`UploadJanitor`, every 10 minutes) deletes the folder once the job is done, failed or cancelled. A job a stop re-queued keeps it, because the retry reads it. Folders no job refers to go once untouched for a day (uploads never queued). |
| Per-chunk clips | No (`VIDEO_CLIPS_ENABLED=false`) | A citation points at the recording with its start and end; the gateway cuts the excerpt (±5 s) when it is opened, cached in the VPS's 5 GB source cache. |
| 16 kHz WAV | No (`KEEP_EXTRACTED_AUDIO=false`) | Only the audio stages read it. Deleted once the recording is indexed; a source that failed keeps it for the retry, and a later run re-extracts it if needed. |
| Original video/audio | A compact copy (`COMPACT_MEDIA_ORIGINALS=true`) | Once indexed, replaced by a 720p H.264/AAC MP4 with `+faststart` (audio: AAC m4a, 64 kbit/s), which every browser plays and seeks without a transcode. |
| Frames, documents, images, database | Yes | Small. |

**What the compact copy changes:**
- Without `PREPARE_VIDEOS`, ingestion reads the full-quality upload and the copy is only what is kept and played back. With it, the box ingests the 720p copy the VPS made (see "Videos are prepared on the VPS"). Either way **the full-quality original is not kept on the server.**
- A copy is refused, and the original kept, unless its length is within 0.5 s (or 0.2%) of the original's, because chunk timestamps index into it, and it is at least 20% smaller (`COMPACT_MIN_SAVING`).
- A file a browser already plays at a modest bitrate (an `.mp4` H.264 at most 720p and 3 Mbit/s, or a lean m4a/mp3) is left alone. YouTube downloads usually are.
- The source keeps the hash of its upload, so uploading the same file again still finds it.
- Tuning: `COMPACT_VIDEO_MAX_HEIGHT` (720, the short side, so portrait videos too), `COMPACT_VIDEO_CRF` (26), `COMPACT_AUDIO_BITRATE_KBPS` (64).
- It costs CPU time after indexing (x264 `medium`, niced), not GPU time.

**Material ingested before these settings.** The backfill needs the files, so it runs only where they are on the same machine as an appliance venv: the local version, or an old box from before the move. The hosted box refuses it (it holds no files, and new ingestions compact as they go). Run it while the ingestion queue is idle:

```bash
set -a; . /workspace/web/api.env; set +a
venv/bin/python -m web_deploy.provision.compact_storage --all --dry-run   # what would be freed
venv/bin/python -m web_deploy.provision.compact_storage --all             # free it (or name firms)
```

Locally, load `web_deploy/local/local.env` instead and add `TENANTS_ROOT=onyx_data/local-web/tenants` and `CONTROL_DATABASE_URL=sqlite:///onyx_data/local-web/control.sqlite3`. Each firm runs in its own child process (`IzvoriaEngine.compact_storage`); the run is refused while an ingestion is running.

## Backups

Everything a firm has lives on the VPS, so the VPS backs it up to a **Hetzner Storage Box** with [restic](https://restic.net): encrypted with `RESTIC_PASSWORD` before it leaves the machine, and deduplicated, so after the first run only what changed travels. The GPU box has nothing to back up.

| What | How | When | Kept |
|---|---|---|---|
| Every firm database and the queue (`pg_dump`, custom format), and every firm's `vault/` and `reports/` | `onyx-web-backup` service (`web_deploy/backup/runner.py`), series `onyx-data` | Every 24 h, **and once an ingestion job has finished**, when the queue is empty | 7 daily, 4 weekly, 6 monthly |
| The gateway database (firms, accounts, sign-up requests) | `onyx-web-gateway-backup.timer` (`web_deploy/backup/gateway.py`), series `onyx-gateway` | Daily at 03:30 | Same |

Not backed up: `uploads/` (their files are in the vault) and the source cache. A failed run is retried after an hour; `/var/lib/onyx-web/backup-status.json` and `journalctl -u onyx-web-backup` say what happened.

**Setting it up:**
1. Order a Storage Box (BX11, 1 TB, is plenty to start) in the same region as the VPS. Enable SSH support.
2. Set `RESTIC_REPOSITORY=sftp:<user>@<user>.your-storagebox.de:onyx-backups` in `/etc/onyx-web-gateway.env`. `RESTIC_PASSWORD` was generated there: **store it off the machine** (a password manager), because without it the backups can't be read.
3. Add the public key `install_gateway.sh` printed (`/etc/onyx-web/backup_ed25519.pub`) to the Storage Box: `ssh-copy-id -p 23 -s -i <key> <user>@<user>.your-storagebox.de`, or its `.ssh/authorized_keys`.
4. Check it: `systemctl restart onyx-web-backup && journalctl -fu onyx-web-backup` (the first backup starts at once), and `systemctl start onyx-web-gateway-backup`. The first run creates the repository.

**Restoring** (`web_deploy/backup/restore.py`), onto a new VPS:
1. Run `install_gateway.sh` with the **same** `/etc/onyx-web-gateway.env` (the database password, restic settings, and the `WEB_API_SECRET` the box knows) and the old backup key, or add the new key to the Storage Box.
2. `systemctl stop onyx-web-gateway onyx-web-backup` (Postgres keeps running), then:
   ```bash
   cd /opt/onyx_v3 && set -a && . /etc/onyx-web-gateway.env && set +a
   /opt/onyx-web-gateway/bin/python -m web_deploy.backup.restore --list
   /opt/onyx-web-gateway/bin/python -m web_deploy.backup.restore --snapshot latest --yes
   chgrp -R izvoria-data /workspace/tenants && chmod -R g+rwX /workspace/tenants
   systemctl start onyx-web-gateway onyx-web-backup
   ```
   It replaces every firm database and vault with the snapshot's. The restore reads the `onyx-data` series by default; `BACKUP_HOST_NAME=onyx-box` reads a backup the GPU box took before the data moved.

The restore was checked end to end against Postgres 15 with pgvector and restic (back up, drop a table and delete a vault file, restore, and both come back), before the move to the VPS.

## Uploading a lot of video

The upload from the firm's office is usually the slowest part of getting material in. At 1 Gbit/s, 10 TB takes about a day at best, and one connection to a server far away rarely fills such a line. Three things make it as fast as the line allows.

**1. Resumable, parallel uploads** (`web_deploy/common/chunked.py`; the web app's `src/api/uploads.ts`):
- Every file goes in 32 MB pieces, four at a time, also within one large file:
  ```
  POST /api/uploads/{batch}/{name}/chunks      {"size": S, "chunk": C}  -> {"received": [...], "complete": false}
  PUT  /api/uploads/{batch}/{name}/chunks/{i}  bytes of piece i         -> {"complete": false}
  ```
- Each piece is written at its offset into one hidden file of the full size, so finishing a file is a rename, never a copy. A marker per piece says it arrived whole.
- A failed piece is sent again after 1, 2, 4… up to 30 s; a refusal (a file too large) is final.
- **An upload broken off continues where it stopped:** the client asks which pieces are there and sends the rest. In the web app, picking the same files again (same kind and language, within 20 hours) continues the same batch.
- Queueing a batch discards anything half-sent, so the box never gets a partial file.
- The old one-request `PUT /api/uploads/{batch}/{name}` still works, for small files and scripts.

Measured on the dev box through the real gateway over loopback: ~4 Gbit/s with four pieces in flight, so the gateway itself is not what limits a 1 Gbit/s line.

**2. The uploader: the 720p copy made at the firm** (`python -m web_deploy.uploader`, `web_deploy/uploader/`). For a firm with raw footage, the biggest win is to send less. Run on the firm's own computer, the uploader makes each video's 720p copy there (the same encode as "Videos are prepared on the VPS", `web_deploy/common/video_copy.py`), so only the copy crosses the internet: raw 4K at ~100 GB an hour becomes about 1 GB, and a day of uploading becomes minutes.
```bash
python -m web_deploy.uploader --server https://onyx.example.com --firm legal-demo --email you@firm.com \
    --mode video --language en "D:/Depositions 2026"
```
- It signs in as the person (asking for the firm password and their own, or reading `IZVORIA_FIRM_PASSWORD` / `IZVORIA_PASSWORD`), then makes the copies one at a time while the previous one uploads (at most two wait on disk).
- Each copy goes in parallel pieces, with retries; `POST /api/uploads/{batch}/record` then tells the gateway what each copy was (its original's hash and name), so duplicate detection and citations work as for the original.
- **Running the same command again after an interruption continues it.** A state file remembers the batch and every finished file, and a half-sent file resumes from the pieces the gateway already has.
- A copy that fails (or an MP4 that is already compact) is sent as it is. Audio, documents and images are never re-encoded.
- Options: `--no-shrink`, `--parallel`, `--chunk-mb`, `--crf`, `--preset`, `--scratch`.
- **Needs** Python 3.10+ and ffmpeg/ffprobe on the PATH; nothing else (standard library only). It runs from a copy of this repository. Packaging it as a single Windows/macOS program is open.
- Checked end to end against a real gateway: two 278 MB 4K test clips became 1.7 MB copies in 4 s each, sent, recorded and queued; a run cut after 50 pieces continued the same batch, sent only the 508 missing pieces, and the stored files were byte-identical.

**3. The receiving side.** A Hetzner cloud VPS's network is shared and not guaranteed; a **dedicated server** has its own 1 Gbit/s port (10 Gbit/s as a paid add-on, useful only if the office line is faster too) and the cores the 720p encode needs if it happens on the VPS instead. Put it in the **Hetzner location nearest the firms** (Falkenstein or Nuremberg for Serbia, rather than Helsinki): lower latency means each connection runs faster.

## Adding videos from YouTube

On the Ingest page, **Video** batches can also come from YouTube links: single videos, Shorts, youtu.be links and whole playlists. This feature is web only. The on-premise appliance never reaches the internet, and this is the one place the web version does.

- **Checking.** "Check links" resolves the links with yt-dlp without downloading anything (`web_deploy/common/youtube.py`). The user sees every video, can drop some, and is told which links added nothing and why (private, deleted, still live, not YouTube).
  - At most `LINK_MAX_VIDEOS` (200) videos go into one batch.
  - A `watch?v=…&list=…` link adds only that video.
- **Queueing.** The box API resolves the links and queues the job with its videos in the job row (`ingest_jobs.manifest`), marked `origin: youtube`. No folder exists yet: the box never writes into the VPS's uploads.
- **Downloading.** When the job reaches the front of the queue, the supervisor writes the manifest into the job's scratch folder and first runs `ingest_worker/fetch_links.py` as a child process:
  - It runs **before the GPU is asked for**, so on dev hardware chat keeps answering while the videos download.
  - Videos are fetched one at a time, up to `LINK_MAX_HEIGHT` (720p), as H.264 + AAC MP4.
  - Each file is named `<title> [<video id>].mp4`; the title is what citations show.
  - A video that fails is listed in the batch's report.
  - If none can be fetched, the job fails before the GPU is touched.
  - A stop re-queues the job; the downloads start again, since the scratch is wiped.
  - The downloaded files are never pushed to the VPS; their copies in the vault are.
- **A rented box needs a signed-in YouTube account.** YouTube answers a data-centre IP that isn't signed in with "Sign in to confirm you're not a bot" for every video (found 2026-10-14 on the Vast box; listing a playlist still works). So the box downloads with the cookies of a **spare** Google account, never a personal one (`gateway/youtube_cookies.py`):
  - **Export them:** open a private window, sign in to the spare account, open youtube.com, export its cookies with a cookies.txt extension (e.g. "Get cookies.txt LOCALLY"), then close the window *without signing out*, so the session isn't rotated away.
  - **Upload them:** in the owner console, GPU server → YouTube downloads, or on the VPS `sudo -u onyx /opt/onyx-web-gateway/bin/python -m web_deploy.gateway.admin_cli set-youtube-cookies cookies.txt`. Only youtube.com and google.com cookies are kept, in `YOUTUBE_COOKIES_FILE` (beside the gateway database, mode 0600). The console shows whether they are signed in and when the sign-in expires.
  - **How they travel:** the VPS staging sends them to the box (`/workspace/web/youtube_cookies.txt`, mode 0600) for a YouTube job only. The supervisor deletes them as soon as the download step ends, and every start deletes any left by a crash. In local mode (one disk) yt-dlp reads the gateway's own file.
  - **They keep themselves alive.** YouTube rotates session cookies as they are used; after each download step the box asks the VPS to take back the jar yt-dlp saved (`/internal/box/youtube-cookies/{job}/refreshed`), which replaces the stored one only if it is still a signed-in session.
  - **Upkeep:** re-export only when YouTube refuses them anyway. The box then reports it (`…/refused`), the owner gets one email a day while it lasts ("Izvoria: YouTube downloads need fresh cookies", to `SUPPORT_TO`), and the console card shows the date. Without any cookies, a failed batch's error also tells the owner to add them.
  - **The risk:** YouTube may flag an account used this way. Today it is izvoria.support@gmail.com, which also sends the request emails, so a lock would stop those too. A dedicated account would avoid that.
- **Turning it off:** `LINK_INGEST_ENABLED=false` hides the panel and refuses the routes.
- **Keeping it working.** yt-dlp (`yt-dlp[default]`, which includes the challenge solver) and deno (the JavaScript runtime it needs) come from `web_deploy/api/requirements.txt`. YouTube breaks old clients within months: if downloads start failing, run `venv/bin/pip install -U "yt-dlp[default]"` on the box. ffmpeg (already provisioned) merges the video and audio streams.

The Ingest page also takes a **whole folder**, picked or dropped, for every kind of file. Files in subfolders are named after their path inside it (`Unit 3/Part I.mp4` becomes `Unit 3 - Part I.mp4`), so two files with the same name can't overwrite each other.

## Tests

```bash
python3.11 -m venv web_deploy/.venv && web_deploy/.venv/bin/pip install -r web_deploy/requirements-dev.txt
web_deploy/.venv/bin/pytest web_deploy/tests                 # gateway, API (fake engine), queue, worker, staging
cd web_deploy/frontend && npm ci && npm test && npm run typecheck
pytest tests                                                   # the app/ changes (appliance venv)
```

Under the appliance venv, a bare `pytest` skips `web_deploy/tests`, which needs the web stack.

### Local end-to-end without Vast

Run the gateway against a box API on this machine (`run_local` does all of this for you):
- `FAKE_BOX_HOST=127.0.0.1` and `GATEWAY_HTTPS_ONLY=0` (so `BOX_LINK=direct`), no `PUBLIC_URL`;
- `TENANTS_ROOT`, `SOURCE_CACHE_ROOT` and `CONTROL_DATABASE_URL` the same as the API's: the gateway receives the uploads and serves the sources.

Open `http://localhost:8000` and sign in as the firm, then as yourself.

The box API needs:
- `api.env`'s hosted settings, with `POSTGRES_PORT` and `POSTGRES_PASSWORD` for the local database and no `BOX_GATEWAY_URL` / `BOX_CREDENTIALS_URL` (one disk, a password in the environment);
- a running `llama-server`.

## Open items

- **Data-handling statement.** "No data leaves the premises" (CLAUDE.md) is true of the appliance, not of this. The hosted product's statement should describe "Where data lives", including its two remaining exposures.
- **A quick check for compression artifacts on slide text** before setting `PREPARE_VIDEOS=true` (see "Videos are prepared on the VPS"); resolution itself is not a concern.
- **The two-machine run.** The 3090 measurements above predate the move. Run the layout end to end (ingest the lecture while three users chat), then on the box check that `find /workspace/tenants -type f` and `ls /workspace/staging-state` are empty and `ss -ltn` shows nothing public, and compare chat latency with the 2026-10-10 run (median 1.6 s). Then stop the box and check that its keys no longer log in to the VPS.
- **The spike's `shared` stage** still assumes the old layout (see "M0 spike").
- **Report speed.** Two reports beside live chats share four slots; each slows the others.
