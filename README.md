# Izvoria

**A private, on-premise AI appliance that answers questions from an organisation's own recordings and documents, and cites a source for every claim.**

Izvoria is built for law firms, hospitals and other businesses whose data may not leave the building. A client points it at their local video, audio, documents and images. Izvoria transcribes, reads and indexes all of it on local hardware, then lets people search it, chat with it, and commission multi-page research reports over it. Every answer comes from a local LLM that may only use the indexed material, and every sentence carries a `[Doc N]` citation that opens the exact clip, page, line range or image it came from.

There is no cloud API anywhere in the inference path. Nothing is sent to a third party.

> **Status: complete.** The appliance (ingestion, retrieval, four query modes, desktop app, English and Serbian) is finished and in use. A hosted multi-firm web version runs the same engine for demos and remote users. Feature-by-feature history is in [`CHANGELOG.md`](CHANGELOG.md).

---

## Why it exists

A lawyer or doctor who acts on a made-up answer is the failure this project is designed to prevent. Two requirements follow, and the whole architecture serves them:

1. **No ungrounded answers.** The model must not state anything that isn't in the client's data. If the material doesn't cover a question, Izvoria says so.
2. **Every claim is traceable.** Each citation resolves to a verbatim excerpt and a playable or viewable source location.

That is why the pipeline does not rely on a single model pass. It has separate correction and verification stages: phonetic correction of transcripts, hallucination correction of visual descriptions, an OCR check on text the vision model claims to read, a scrubber that drops weak retrieved evidence, and citation validation on generated answers and reports.

## What it does

### Ingest anything a firm keeps

| Source | How it is read | Cited as |
|:--|:--|:--|
| **Video and audio** | VAD → domain "scout" pass → Whisper large-v3 transcription with domain hotwords → speaker diarization → sentence reconstruction → glossary-based phonetic correction. For video, a visual branch finds key moments and describes the frames with a vision-language model, grounded in what the speaker is saying. | timestamp + speaker, opens the clip |
| **PDF, DOCX** | Split page by page, converted with MarkItDown; tables re-read with a custom layout extractor (ruled and borderless); scanned pages OCR'd with RapidOCR, with lines OCR missed counted and reported. | page + heading |
| **TXT, Markdown, EPUB, HTML, RTF** | Cut along the file's own chapters, clauses and headings. | line range or section |
| **Images** | Described by Qwen2-VL; any text it claims to see is kept only if OCR read it too. | image |

Every chunk is placed in context by a small local model (contextual retrieval), embedded with BGE-M3 and stored in PostgreSQL + pgvector next to a full-text index.

### Ask it four ways

| Mode | What you get |
|:--|:--|
| **Search** (standard / verified) | Hybrid dense + keyword search, fused with RRF and re-ranked by a cross-encoder. Verified search also drops evidence a scrubber model judges unreliable. |
| **Chat** | Multi-turn cited answers from Qwen3.5-9B with a 128K context. `[Doc N]` numbers stay stable for the whole conversation, so a follow-up can cite an earlier turn's evidence. |
| **Reasoned answer** | Several retrieval passes judged by a reasoning model, then a draft and a self-critique before the answer. |
| **Research report** | Topic in, multi-page cited report out. Izvoria proposes an outline for you to edit and approve, sweeps the corpus into a numbered evidence bank, and writes section by section, each section seeing only its own evidence. An exhaustive mode keeps gathering while sections remain unsupported. Sections the material can't support are listed as *not covered* instead of being filled with uncited text. Exports to Markdown and a typeset PDF (formulas, tables, bookmarks). |

### Two languages in one index

English and Serbian (Latin and Cyrillic) live in one database. Each chunk is keyword-indexed with its own stemmer, and the multilingual dense leg finds evidence across languages. A Serbian question about English material returns an answer in Serbian that quotes the English excerpt verbatim. Nothing is ever translated, because a translated index would put model-written text under every citation. Every language-dependent rule (stopwords, refusal text, stemmer, Whisper code, model choices) lives in a validated language pack (`data/languages/*.toml`), so adding a language doesn't touch code.

### Three ways to run it

- **Desktop app** (PySide6 / Qt 6): Chat, Search, Ingest and Report views, with clickable citations, an outline-approval step, progress, cancel and history. It runs the engine in-process.
- **CLI** (`python -m app.main`) over the same engine.
- **Hosted multi-firm web app** (`web_deploy/`): a React + TypeScript front end and an always-on Hetzner gateway that keeps every firm's data, two-step sign-in, firm administrators, resumable parallel uploads, YouTube and folder ingestion, and encrypted off-site backups. A rented data-centre GPU is started on demand. It holds no client data at rest, reaches the data through an SSH tunnel, and serves every firm from one process: one engine per firm, shared pinned models, and a patched `llama-server` that serves the LLM roles to concurrent users.

---

## Engineering highlights

**Six-plus models on a 12 GB card.** The target hardware is two RTX 3060s; development happens on one. No GPU here can hold more than one or two of the pipeline's models, so nothing assumes a model stays resident. Every GPU component goes through one orchestrator (`app/core/model_manager`) that:
- loads models just in time and keeps a **warm pool** budgeted by each model's *measured* VRAM footprint. When a load needs room, it evicts the idle set that frees the fewest megabytes, least recently used on a tie;
- learns each model's peak activation memory with a telemetry thread and uses it for admission control (`OutOfVramError` before a crash, not after);
- preloads the next model on a background worker while I/O-bound work continues, and guards against the subtle ways an unresolved preload can deadlock a long-running process.

A full research report runs in about 6 model loads instead of the ~40 a naive per-section design would need. An integration test counts the loads, so a regression fails the build.

**Hexagonal, feature-first architecture.** Each of the 30+ features (`app/core`, `app/features`, …) has the same ports-and-adapters layout: pure `domain/` interfaces, `data/` adapters for models and repositories, one `service/` API as the only entry point, and a typed exception hierarchy with an error mapper at the boundary. `IzvoriaBootstrapper` is the DI composition root. Orchestration `Phase`s wrap feature APIs with job tracking and model lifecycle, and `Pipeline`s chain them.

**Grounding you can measure.** Evaluation scripts (`scripts/eval/`) download real speech (FLEURS, ParlaSpeech-RS, lectures), ingest it into isolated databases and report word accuracy, language-guard behaviour, retrieval hit@k and answer quality, within and across languages. In cross-language runs, unanswerable questions were refused 10/10 and every report citation came from the right material. These runs also found a Whisper batching bug that silently dropped up to 12% of speech in some files (fixed; now under 0.1%).

**Production concerns, not just a demo.** Alembic migrations applied on start; per-thread preload slots and a strict-residency mode for concurrent requests; a short-prompt-first patch to `llama-server` so one long report section doesn't stall every chat; compact 720p storage for hosted recordings; restic backups; a GPU box that holds no database password and fetches a per-start credential from the gateway.

## By the numbers

| | |
|:--|:--|
| Application code (`app/`) | ~43,000 lines of Python |
| Desktop app, web version, deploy tooling | ~28,000 lines of Python + ~5,600 lines of TypeScript/React |
| Tests | ~3,000 test functions in ~380 files: mocked unit tests that mirror `app/` exactly, integration tests on real Postgres and GPU, headless Qt tests, gateway/API tests |
| Local models orchestrated | ~15 (ASR, VAD, diarization, 3 LLM families, VLM, embedder, reranker, OCR, …) |
| Feature write-ups | one per module in [`docs/`](docs/) |

## Tech stack

| Area | Tools |
|:--|:--|
| Speech | faster-whisper (large-v3), NVIDIA NeMo (MarbleNet VAD, TitaNet diarization), wtpsplit (SaT) |
| LLMs | llama.cpp / `llama-cpp-python` and `llama-server` (GGUF): Qwen3.5-9B, Qwen2.5, Llama 3.2 3B, Gemma 4 E4B |
| Vision | Qwen2-VL-7B on Hugging Face Transformers + bitsandbytes 4-bit |
| Retrieval | BGE-M3 embeddings, bge-reranker-v2-m3 cross-encoder, PostgreSQL 15 + pgvector + full-text search, RRF fusion |
| Documents | MarkItDown, pdfplumber, pypdfium2, RapidOCR (PP-OCRv5, ONNX), striprtf, charset-normalizer |
| Reports | fpdf2, ziamath (LaTeX → SVG), markdown-it-py |
| Backend | Python 3.11, SQLAlchemy, Alembic, pydantic-settings, PyTorch |
| Desktop | PySide6 (Qt 6) |
| Web | React, Vite, TypeScript, Starlette, Caddy, SQLite, restic, Vast.ai, Hetzner |
| Testing | pytest, unittest.mock, Dockerised Postgres, Vitest |

## Repository map

```
app/
├── core/               model manager (VRAM orchestration), database, jobs, language packs, anchors
├── features/           one hexagonal module per pipeline stage or capability
├── orchestration/      Phases (feature + job + model lifecycle) and Pipelines (ingest, search, chat, reports)
├── engine/             IzvoriaBootstrapper (DI root) and IzvoriaEngine (top-level facade)
└── main.py             CLI
frontend/               PySide6 desktop app
web_deploy/             hosted multi-firm web version (gateway, API, ingestion worker, React app, backups)
deploy/                 single-tester hosted demo of the desktop app over noVNC
data/                   language packs, glossaries, fonts, migrations
scripts/eval/           ASR, retrieval and cross-language benchmarks
tests/ · integration_tests/ · frontend/tests/
docs/                   one write-up per module
```

Start with [`CLAUDE.md`](CLAUDE.md) for the architecture and conventions, [`docs/`](docs/) for each module, and [`CHANGELOG.md`](CHANGELOG.md) for how it was built.

---

## Setup

There are three ways to run Izvoria:

- **A. Local appliance / development machine.** Izvoria runs on a Linux box with an NVIDIA GPU (≥12 GB VRAM), which is how the product ships.
- **B. Hosted demo.** A tester uses the real desktop app from a browser. An always-on gateway VPS starts a rented Vast.ai GPU on demand. Your own PC doesn't need to be running.
- **C. Hosted multi-firm web version.** Many firms and users on one web app, with firm data on a Hetzner VPS and a stateless rented GPU. It can also run entirely on the 12 GB dev card for local testing. The full guide is [`docs/web_deploy.md`](docs/web_deploy.md); folder overview in [`web_deploy/README.md`](web_deploy/README.md).

### **A. Local setup (Debian / Ubuntu, NVIDIA GPU)**

#### 1. Prerequisites

- An NVIDIA GPU with ≥12 GB VRAM and the proprietary driver installed (`nvidia-smi` works). `update_nvidia_driver.sh` updates it on Debian.
- `git`, `ffmpeg`, `curl`, and the build tools for Python (`build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev`).

#### 2. Environment (Python, venv, CUDA, databases)

```bash
git clone git@github.com:StefanovicAleksa/onyx_v3.git
cd onyx_v3
./setup_env.sh
```

`setup_env.sh` is idempotent, so it's safe to re-run. It:

1. builds Python 3.11.9 in user space;
2. creates `venv/` and installs `requirements.txt`;
3. installs the CUDA compiler and rebuilds `llama-cpp-python` with GPU support;
4. installs Docker;
5. starts both Postgres + pgvector containers from `docker-compose.yaml`: `db_dev` on port 5432 and `db_test` on port 5433.

Then install the packages that must skip dependency resolution. `rapidocr` declares `opencv-python`, which would overwrite the headless OpenCV that the desktop app needs:

```bash
venv/bin/pip install --no-deps -r requirements-nodeps.txt
```

#### 3. Configuration

```bash
cp .env.example .env
```

Fill in `.env`:

| Key | Value |
|:--|:--|
| `ENV` | `development` (`.env`), `test` (`.env.test`) or `production` (`.env.prod`); `ENV` selects which file is read |
| `POSTGRES_SERVER` / `POSTGRES_PORT` | `localhost` / `5432` |
| `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` | must match `db_dev` in `docker-compose.yaml` |
| `APPLIANCE_LANGUAGE` | optional: `en` or `sr`, the language the app starts in (Slovenian is a legacy language, not served). One database holds every language; the app switches the answer language at runtime and starts in the one it was last switched to unless this is set |
| `APPLIANCE_INDUSTRY` | optional glossary name under `data/glossaries/<language>/` |

`.env.test` holds the same keys for `db_test` (port 5433), and is needed only for integration tests.

#### 4. Models

Every model loads from `models/`. Nothing is downloaded at runtime, and a missing model fails the phase that needs it. Two download scripts exist:

```bash
./models_download.sh                           # Qwen3.5-9B Q4_K_M (chat, reasoning, reports)
venv/bin/python -m scripts.download_ocr_models  # RapidOCR PP-OCRv5 (scanned PDFs)
```

The rest come from an existing Izvoria model set, for example `rsync` from the dev machine. For English and Serbian, that is:

| Path under `models/` | Used for |
|:--|:--|
| `vad_multilingual_marblenet.nemo` | voice activity detection |
| `whisper-tiny-en` | English domain scout |
| `faster-whisper-large-v3` | transcription (and the Serbian scout) |
| `titanet_large.nemo` | speaker diarization |
| `fullstop-punctuation-multilingual-base` | punctuation |
| `llama-3.2-3b-instruct-q8_0.gguf` | English classification, context, ASR correction |
| `google_gemma-4-E4B-it-Q4_K_M.gguf` | Serbian replacement for the 3B models |
| `qwen2.5-3b-instruct-q4_k_m.gguf`, `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | phonetic/visual correction, visual cues, retrieval scrubbing |
| `Phi-3.5-mini-instruct`, `flanec-base-cd`, `deberta-v3-base-zeroshot` | NLP, ASR correction, hallucination scrubbing |
| `qwen2-vl-7b-hf-bnb4bit` | frame and image description (Hugging Face Transformers, not GGUF) |
| `bge-m3`, `bge-reranker-v2-m3` | embeddings and reranking |
| `Qwen3.5-9B-Q4_K_M.gguf` | QA generation, reasoner, reports |
| `rapidocr/` | OCR |

That's about 51 GB. To regenerate the list after adding a model, run `grep -rhoE "[\"']models/[^\"']+[\"']" app data | sort -u`.

#### 5. Run it

```bash
docker compose up -d db_dev                            # if it isn't already up

venv/bin/python frontend/main.py                       # desktop app; click "Start appliance"
venv/bin/python frontend/main.py --autostart           # ...or boot it without the click
venv/bin/python frontend/main.py --stub                # UI only: no GPU, no database, canned answers
venv/bin/python -m app.main                            # CLI menu over the same engine
```

The engine applies database migrations itself on start (`IzvoriaEngine.start()` → Alembic).

#### 6. Tests

```bash
venv/bin/pytest tests                                                   # unit, fully mocked
QT_QPA_PLATFORM=offscreen venv/bin/pytest frontend/tests                # desktop app, headless
docker compose up -d db_test && venv/bin/pytest integration_tests -m integration   # real DB / GPU

python3.11 -m venv deploy/.venv && deploy/.venv/bin/pip install -r deploy/requirements-dev.txt
deploy/.venv/bin/pytest deploy/tests                                    # hosted-demo gateway + agent
```

### **B. Hosted demo (gateway VPS + on-demand Vast.ai GPU)**

This costs about $35–60/month instead of $200+ for a dedicated GPU server. The GPU box is billed only while it's started, plus its disk.

1. **Rent the GPU box.** On Vast.ai, rent one RTX 3090 with ≥120 GB of disk. Use image `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, ports `-p 8090:8090 -p 6080:6080`, and on-start script `bash /workspace/onyx_v3/deploy/instance/onstart.sh`.
2. **Copy Izvoria and the models to it.** From a machine that has the models:
   ```bash
   deploy/instance/sync_to_box.sh root@<ssh-host> <ssh-port>
   ssh -p <ssh-port> root@<ssh-host> bash /workspace/onyx_v3/deploy/instance/provision.sh
   ssh -p <ssh-port> root@<ssh-host> bash /workspace/onyx_v3/deploy/instance/onstart.sh
   ```
   `provision.sh` prints `AGENT_SECRET` and `WEBSOCKIFY_TOKEN`; keep them for step 4.
3. **Load the demo material.** Ingest it one batch per language, with the Ingest view's language set for each (all into the one `onyx` database), then stop the instance.
4. **Set up the gateway.** On a ~$5/mo VPS with your domain's A record pointing at it:
   ```bash
   git clone git@github.com:StefanovicAleksa/onyx_v3.git /opt/onyx_v3
   sudo bash /opt/onyx_v3/deploy/gateway/install.sh demo.example.com
   sudoedit /etc/onyx-gateway.env      # password, AGENT_SECRET, WEBSOCKIFY_TOKEN, VAST_API_KEY, VAST_INSTANCE_ID
   sudo systemctl restart onyx-gateway
   ```
5. **Send the tester the URL and the password.** They sign in, pick Serbian or English, click **Start appliance** (about 1–2 minutes), upload files and open the desktop. The box stops itself after 30 idle minutes, but never mid-ingestion or mid-report.

Once it's set up, your own PC doesn't need to be on. It's only needed to push code updates, or to re-upload the models if the Vast instance is destroyed, so keep a backup of `models/` somewhere else. The full runbook, the costs and the known risks are in [`docs/deploy.md`](docs/deploy.md). A stopped Vast instance can lose its GPU to another renter, and demo data sits on rented hardware, not the client's premises.

### **C. Hosted multi-firm web version**

See [`docs/web_deploy.md`](docs/web_deploy.md): provisioning the Hetzner gateway and the Vast.ai box, firm and account administration, backups and restore, and `web_deploy/local/run_local.py` for running the whole web version on one dev GPU. Its tests run with `web_deploy/.venv/bin/pytest web_deploy/tests`.

---

## Author

Built by **Aleksa Stefanović** ([GitHub](https://github.com/StefanovicAleksa)).
