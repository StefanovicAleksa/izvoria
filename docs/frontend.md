# Frontend (`frontend/`)

The desktop application. PySide6 (Qt6) widgets driving `OnyxEngine` in-process on a background
thread. See `frontend/README.md` for how to run it and its known limits; this document covers how
it fits the rest of the system.

## Why it exists

Until now the appliance was operable only through `app/main.py`'s numbered menu.
The product's central claim is that every answer is traceable to its source, and a terminal cannot
deliver that: it prints a path and a timestamp and leaves the user to go find the video. The
desktop app closes that loop — click a citation, read the verbatim words it came from, and
open the recording if you want to hear them.

## Where it attaches

Only at `OnyxEngine`. The app never imports a feature API, a Phase or a pipeline directly, and the
composition root stays `OnyxBootstrapper.build()`.

```
frontend/onyx_desktop/data/engine_gateway.py   ← the only module importing app.*
        │
        └── OnyxBootstrapper.build(progress_listener=…) → IOnyxEngine
                 start / stop / get_state
                 search_standard / search_verified
                 chat / chat_reasoned / research reports
                 ingest_videos / ingest_audios
                 get_chunks / list_sources        ← added for this app
```

There is deliberately **no embedded media player**. Source files are handed to the desktop's own
player via `QDesktopServices.openUrl`, which is why the app needs only `PySide6-Essentials` and why
Qt's multimedia stack — and its hardware-decode crash on NVIDIA hardware without a usable VAAPI
driver — is not in the process at all. The cost is that a system player cannot be told to start at a
timestamp, so a citation into a whole recording shows the cited span prominently instead.

Everything above the gateway speaks in frozen dataclasses from
`frontend/onyx_desktop/domain/entities.py`. No pydantic entity and no `app.*` type crosses onto the
GUI thread, which is what lets the entire UI run and be tested against a stub gateway with no GPU,
database or model weights present.

## What it added to `app/`

Two small, additive changes, both defaulted so the CLI is unaffected:

- **`OnyxEngine.get_chunks(chunk_ids)` / `list_sources()`** (`app/engine/service/onyx_engine.py`,
  declared on `IOnyxEngine`). A `QACitation` carries ids, timestamps and a resolved path but **no
  text**, so rendering the cited excerpt beside its clip requires reading the chunks back.
  `get_chunks` is a plain indexed read — no model, no VRAM — and, unlike `RetrievalAPI.get_chunks`,
  the facade resolves physical paths onto the results the same way a search result gets them.
- **`JobManager(repository, progress_listener=None)`** (`app/core/jobs/service/manager.py`), fired
  at start/finish/fail of every job, and passed through by
  `OnyxBootstrapper.build(progress_listener=…)`. Every Phase already wraps its per-file work in
  `run_job`, so this single hook observes every file of every phase in every pipeline. It is the
  only way a long-running caller can see batch progress, because the pipelines report nothing until
  `process_directory` returns. A listener that raises is logged and ignored — observation must never
  fail the work it observes.

## Engine behaviours the UI has to model

These are properties of the engine, not choices of the UI:

| Behaviour | Consequence for the app |
|---|---|
| `physical_path` is injected with `object.__setattr__`, not declared | always read it with `getattr(x, 'physical_path', None)` |
| A citation's path is the extracted clip **or** the whole source file | the panel says which, and shows the timestamp to skip to in the second case |
| `QACitation` has no text | excerpts come from a separate `get_chunks` read |
| `generated_answer` is the raw model text while `citations` was parsed from a normalized copy | the gateway re-runs `normalize_citation_tags` so `[Doc N]` links are reliable |
| Reasoned Answer returns an answer **or** a clarification question | clarification is a first-class turn type, and re-asking appends `[Clarification N]` to the original question because the reasoner cannot resume mid-loop |
| Ingestion and live inference are mutually exclusive | live views are disabled for the whole ingestion |
| The VRAM pool holds one model at a time | every engine call is serialized onto one worker thread |
| Nothing is cancellable | no Cancel button; closing mid-ingestion warns before quitting |

## Related documents

- `docs/engine.md` — the facade and bootstrapper this app talks to.
- `docs/orchestration.md` — the Phases and pipelines behind each call.
- `docs/core/jobs.md` — the job wrapper the progress listener hooks into.
- `docs/features/clip_extractor.md` — why per-chunk clips exist, i.e. what a citation usually opens.
