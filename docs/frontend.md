# Frontend (`frontend/`)

The desktop application. PySide6 (Qt6) widgets driving `IzvoriaEngine` in-process on a background
thread. See `frontend/README.md` for how to run it and its known limits; this document covers how
it fits the rest of the system.

## Why it exists

Until now the appliance was operable only through `app/main.py`'s numbered menu.
The product's central claim is that every answer is traceable to its source, and a terminal cannot
deliver that: it prints a path and a timestamp and leaves the user to go find the video. The
desktop app closes that loop — click a citation, read the verbatim words it came from, and
open the recording if you want to hear them.

## Where it attaches

Only at `IzvoriaEngine`. The app never imports a feature API, a Phase or a pipeline directly, and the
composition root stays `IzvoriaBootstrapper.build()`.

```
frontend/izvoria_desktop/data/engine_gateway.py   ← the only module importing app.*
        │
        └── IzvoriaBootstrapper.build(progress_listener=…) → IIzvoriaEngine
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

On the hosted demo box the agent launches the app with `ONYX_SOURCE_OFFER_URL`/`ONYX_SOURCE_OFFER_TOKEN`, and
`system_open.open_source` posts the citation to the agent instead (`data/source_offer_client.py`, stdlib only), so
the source reaches the tester's browser with sound, at its page or lines, with a download. Without those variables
nothing changes. See "Sources in your browser" in `docs/deploy.md`.

Everything above the gateway speaks in frozen dataclasses from
`frontend/izvoria_desktop/domain/entities.py`. No pydantic entity and no `app.*` type crosses onto the
GUI thread, which is what lets the entire UI run and be tested against a stub gateway with no GPU,
database or model weights present.

## What it added to `app/`

Two small, additive changes, both defaulted so the CLI is unaffected:

- **`IzvoriaEngine.get_chunks(chunk_ids)` / `list_sources()`** (`app/engine/service/izvoria_engine.py`,
  declared on `IIzvoriaEngine`). A `QACitation` carries ids, timestamps and a resolved path but **no
  text**, so rendering the cited excerpt beside its clip requires reading the chunks back.
  `get_chunks` is a plain indexed read — no model, no VRAM — and, unlike `RetrievalAPI.get_chunks`,
  the facade resolves physical paths onto the results the same way a search result gets them.
- **`JobManager(repository, progress_listener=None)`** (`app/core/jobs/service/manager.py`), fired
  at start/finish/fail of every job, and passed through by
  `IzvoriaBootstrapper.build(progress_listener=…)`. Every Phase already wraps its per-file work in
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
| Switching language rebuilds the engine | "Answer in" (sidebar) is a worker job like any other: refused while busy, and history reads are closed for its few seconds, because the database pool is replaced |
| A batch, or a resumed report, can change the active language | the worker reports the active language after every boot, ingest, resume and switch (`activeLanguage`), and the bridge re-emits changes as `languageChanged`, so every picker and header follows the engine |
| The VRAM pool holds one model at a time | every engine call is serialized onto one worker thread, except two: cancelling a report (GUI thread) and history (reader thread), neither of which touches a model |
| Nothing is cancellable | no Cancel button; closing mid-ingestion warns before quitting |

## History

Chat, Search and Report each have a history pane on the left (`ui/widgets/history_list.py`), filled once the engine has started. History lives in the appliance database, not in a GUI file. A reinstalled desktop, the hosted demo's per-language databases and a reset all see the right history without drifting from the data. (`data/session_store.py`, the old JSON index at `~/.config/onyx-desktop/sessions.json`, was written but never read back, and is gone.)

- **Chat**: reopening a conversation re-renders every turn with clickable evidence under its original `[Doc N]` numbers, and the next question continues the same session. Switching is ignored while that view's own answer is pending, so a late reply can't land in another conversation's transcript. Deleting the open conversation starts a new one.
- **Search**: reopening shows the hits it returned then, at their original ranks and scores, under a "Saved results from …" banner. **Run again** repeats it against today's index.
- **Report** (no delete: reports are deliverables):
  - finished → opens;
  - outline never approved → back to the editable approval step;
  - stopped, failed, or interrupted mid-run (still listed as gathering/writing when the app went away) with an evidence bank → the written sections plus **Resume** (`IzvoriaEngine.resume_research_report`), which writes only the missing sections and gathers nothing, so written citations keep their numbers;
  - stopped before any evidence → says so and prefills the topic.

  The pane won't navigate away from a running report.

**Reads bypass the worker.** `EngineBridge` runs history calls on a second `QThread` (`service/history_reader.py`, thread `izvoria-history`). An ingestion or report holds the worker for hours, and picking up an old conversation must not wait behind it. It's safe because `HistoryReader` refuses any gateway method but the database-only history ones (`HISTORY_CALLS`). These reads never touch `busyChanged` or the status bar, and deletes re-list their pane themselves. Resume is the opposite: a GPU job on the worker that takes `RESEARCHING` like any report run.

## Related documents

- `docs/engine.md` — the facade and bootstrapper this app talks to.
- `docs/orchestration.md` — the Phases and pipelines behind each call.
- `docs/core/jobs.md` — the job wrapper the progress listener hooks into.
- `docs/features/clip_extractor.md` — why per-chunk clips exist, i.e. what a citation usually opens.
