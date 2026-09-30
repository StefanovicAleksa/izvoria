# Onyx Desktop

A native PySide6 (Qt6) desktop app for the Onyx appliance. It drives `OnyxEngine`
**in-process** — no HTTP server, no IPC, no second runtime to vendor onto an air-gapped box.

Its reason for existing is the verification loop the CLI cannot offer: read a claim, click its
citation, and see the verbatim words it came from — with the source file one click from opening.

## Running it

```bash
# UI only — no GPU, no database, no model weights. Canned answers, instant startup.
venv/bin/python frontend/main.py --stub

# The real appliance (needs the database up)
docker compose up -d db_dev
venv/bin/python frontend/main.py
```

The real engine does **not** boot on launch. The window opens immediately and you click
**Start appliance** in the sidebar, so a failed boot is a visible status rather than a crash
before anything is drawn. `--autostart` does that click for you once the window is up (the hosted
demo launches the app this way; see `docs/deploy.md`) — a failed boot still shows as a status.

**Stop appliance** (same place, once it runs) shuts the engine down and frees the GPU without closing
the window; **Start appliance** brings it back. It is disabled while anything runs — a chat turn, a
search, an ingestion, a report, a language switch — because each of those holds the engine it would
tear down, and ingestion has no safe stop point.

One visual design, in light and dark: warm neutrals, serif headings, near-square corners, outlined
citation tags and an ink sidebar in both modes. Light/dark is switched with the sun/moon button next to
the wordmark, at any time (even mid-ingestion, since it's pure presentation), and remembered per user
in QSettings. At launch the mode comes from `--mode`, else the remembered choice, else `ONYX_MODE`,
else dark.

The palettes and the stylesheet live in `ui/theme.py`. A switch re-applies the stylesheet, which
repaints almost everything. A widget that bakes a colour into rich text or a pixmap while it builds
(message bubbles' citation spans, the report document, the status dots, the mode toggle's icon) exposes `apply_theme()`, and `MainWindow.set_mode()` calls it on every such widget.
**A new widget that reads `theme.X` into its content needs the same hook**, or it keeps the old
mode's colour after a switch. Prefer a stylesheet role (`setProperty("role", ...)`) when one will do.

### Display language

The app's own text (labels, buttons, messages, step names, dates) can be read in English or Serbian, switched
with the **EN / SR** button next to the wordmark. Like light/dark it applies instantly, even mid-job, and is
remembered per user; `ONYX_DISPLAY_LANGUAGE` picks it when nothing is remembered. It is **separate from
"Answer in"**, which is the appliance's language: someone can read the app in Serbian and have it answer in
English. Answers, excerpts, report text and file names are never translated.

- **Write English, wrap it.** Every on-screen string goes through `utils/i18n.py`: `tr("Start appliance")`,
  `tr("Step {current} of {total}", ...)`, `tr_n(n, "{n} file", "{n} files")` for plurals, and `N_("...")`
  to mark a string in a table that is translated where shown. The English text is the key.
- **Set text through `ui/translate.py`,** not `setText()`: `tl.label("Chat")`, `tl.text(widget, ...)`,
  `tl.tooltip(...)`, `tl.items(combo, [...])`, or `tl.on_retranslate(...)` for content re-rendered from
  data. Each call records how to produce its text and replays it on a switch, which is what makes the switch
  instant without per-view refresh code. A bound widget's text set with a plain `setText()` would be
  overwritten by the next switch.
- **A catalog per language** in `onyx_desktop/locales/<code>.toml` (`[meta]` code, native name, plural rule;
  `[strings]` English → translation). A missing string falls back to English at runtime, but
  `tests/utils/test_i18n_catalogs_unit.py` reads the source for every string and fails on a missing, stale or
  placeholder-breaking entry, and on a served language pack (`data/languages/*.toml`) with no catalog.
  Adding a language: copy `sr.toml`, translate, set `plural` (`english` or `slavic`), run the tests.

### What the screen does not say

The app shows people what the appliance did, not how. Keep it that way when adding UI text:

- **No internals by name.** No models, GPU/VRAM, pipeline stages, job names, retrieval scores, chunk
  counts, storage paths or engine log lines. Engine errors are mapped to plain sentences in
  `utils/error_mapper.py` (and the per-file ones by `file_problem`); the detail goes to the log.
- **Progress is general steps.** Ingestion shows *Preparing files → Analyzing audio → Contextualizing →
  Analyzing visuals → Indexing* (fewer for audio, documents and images), each with its own bar; a
  report shows *Planning → Gathering sources → Organizing → Writing section N → Finishing*.
  `utils/progress_stages.py` is the only place that knows which engine job belongs to which step; a new
  job must be added there, and a job it doesn't know is not shown at all. A step's bar is an estimate
  (files done × jobs in the step) and is marked done outright when the next step starts.
- The one exception is trust: an image's excerpt is still labelled an automatic description, because
  it is machine-written and has to be checked against the picture.

`--stub` is the loop to develop in. A real chat turn costs tens of seconds and a boot costs
seconds more; the stub answers straight away and deliberately cycles through every branch the UI
has to render — a grounded answer, an answer with *no* evidence, a Thorough answer (the engine's Reasoned Answer) clarification
request, an unresolved clip path, and a failure (ask it anything containing "fail").

## Layout

Follows the project's hexagonal convention, plus a Qt-only `ui/` layer:

```
onyx_desktop/
├── domain/      frozen dataclasses, enums, and IEngineGateway — no Qt, no app.*
├── data/        engine_gateway.py (the ONLY module importing app.*), stub_gateway, report_writer
├── service/     engine_bridge / engine_worker (threading), log_relay, desktop_app
├── ui/          theme, icons, assets/, main_window, views/, widgets/
├── exceptions/  DesktopError and friends
└── utils/       error_mapper, formatting
```

The dependency rule that makes the stub possible: **`ui/` and `service/` never import `app.*`.**
Only `data/engine_gateway.py` does, and it maps engine entities into the frozen dataclasses in
`domain/entities.py`. That is also why the whole UI is testable without a GPU.

## Views

Chat, Search, Ingest, **Report**, plus two language pickers. Both list the language packs by their own names ("Srpski"), and both rebuild the value from `currentData()` at the boundary (the QComboBox gotcha):

- **Answer in** (sidebar) switches the active language: the language answers, reports and verified searches are written in. It rebuilds the engine on the worker (seconds, no model load), is enabled only once the appliance runs and while it's idle, and puts itself back if the bridge refuses. Chat, Search and Report show "Answers in …" in their headers.
- **Language of these files** (Ingest view) is the batch's language. The gateway switches to it before ingesting, so after the batch the appliance answers in it too, and the sidebar follows. It defaults to the active language.

The Report view is the odd one out, deliberately:

- It is the only view whose work takes tens of minutes and holds the GPU, so it has a **determinate**
  progress bar (a plan states its step count up front, unlike ingestion) and a **Stop** button.
- It is the only place `[Doc N]` citations are actually clickable. The chat view renders them as
  coloured `<span>`s; you verify a chat claim through the evidence card beneath it. A report is pages
  long and has no cards, so the citation itself is the control — `QTextBrowser` + `setOpenLinks(False)`
  + `anchorClicked`, against the `onyx://doc/N` links the engine's renderer emits in link mode.
- Its **cancel bypasses `EngineBridge`'s worker queue**. That is the single exception to "every engine
  call goes through the worker", and it exists because the worker is busy running the very report being
  cancelled — a queued call would arrive exactly when it is useless. Safe only because the engine just
  sets a `threading.Event`.
- Adding a fifth view means editing **two aligned tuples** in `main_window.py`: the nav button id *is*
  the `QStackedWidget` index.

## Threading

One `QThread` holds the engine. `EngineBridge` (GUI thread) talks to `EngineWorker` (engine
thread) through queued signals, so Qt's event loop serializes every request — one engine call at
a time.

That serialization is a requirement, not tidiness: `acquire_model` evicts every other resident
model, so the VRAM pool holds one model at a time and two overlapping GPU calls would collide
rather than queue. The bridge therefore **refuses** a second request while one is in flight
instead of enqueuing it, and `busyChanged` disables the composer so a double-click cannot try.

Two non-obvious things in here, both of which caused real hangs before they were fixed:

- `worker.finished → thread.quit` is a **DirectConnection**. A `QThread` object's affinity is the
  GUI thread, so an auto connection would queue `quit()` behind a GUI event loop that is blocked
  in `thread.wait()` — a deadlock until timeout.
- `closeEvent` **ignores** the close, then finishes shutting down from a `QTimer.singleShot(0, …)`.
  Calling `self.close()` from inside `closeEvent` is reentrant and leaves the window half-closed,
  and the event loop never exits.

## Known limits

- **Ingestion cannot be cancelled.** Nothing in the pipeline has a cooperative stop point, and the
  thread sits inside native transcription and LLM calls that ignore interruption. The Cancel
  button is deliberately disabled, with a tooltip that just says a batch cannot be cancelled. Closing the window mid-ingestion asks
  whether to quit anyway and warns that partially indexed material may be left behind.
- **Ingestion progress is coarse by design.** It comes from a `JobManager` callback that fires once
  per file per job, folded into a few general steps (see *What the screen does not say*). The
  longest jobs report only when a file finishes, so a step's bar can sit still for minutes on large
  files; after 90 s without a tick the view says it is still working rather than looking hung. The
  engine's log is still captured by `LogRelay` for diagnostics, but is not shown.
- **Conversation history is not re-displayed.** Reopening a conversation sends the engine the same
  session id, so the answer model still reads the earlier turns — but the transcript starts empty,
  because `get_history` is capped at `CHAT_HISTORY_MAX_TURNS` and the engine cannot list sessions
  at all. The conversation index lives in `~/.config/onyx-desktop/sessions.json`.
- **No embedded player, by design.** Onyx is not a media player. An evidence card's **Open clip**
  button hands the file straight to
  your desktop's default player in one click (`xdg-open`; mpv here), which removed a whole class of codec and
  hardware-decode crash — Qt's FFmpeg backend tried VAAPI on this NVIDIA box, failed, and took the
  process down. The tradeoff: a system player cannot be told to start at a timestamp. For extracted
  clips that does not matter (the clip *is* the excerpt); for audio-only sources the panel shows the
  cited span in large copyable text so you can skip to it.
- **No library view yet.** `list_sources()` is wired through the engine but nothing displays it.

## Tests

```bash
QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest frontend/tests
```

Bare `pytest` also collects them (`pytest.ini`'s `testpaths`). GUI tests run headless under the
offscreen platform, so no display is needed and nothing skips.

Two conventions here differ from `tests/` for concrete reasons:

- **No `__init__.py` files.** `frontend/tests/__init__.py` would create a package literally named
  `tests`, shadowing the repo's root `tests` package and breaking collection of the entire existing
  suite. Test filenames therefore carry a `_unit` suffix to stay globally unique — the repo already
  has 29 files named `test_error_mapper.py`, and a plain one here would clash.
- **`conftest.py` overwrites `QT_QPA_PLATFORM`** rather than defaulting it, because this machine's
  shell exports `wayland;xcb`, and the platform plugin is bound when the first `QApplication` is
  constructed.

Gateway tests build **real** pydantic entities rather than `MagicMock`s. A `MagicMock`
auto-creates attributes, so `getattr(chunk, "physical_path", None)` would return a mock instead of
`None` and the unresolved-path branch could never fail a test.
