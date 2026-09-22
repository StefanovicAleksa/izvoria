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
before anything is drawn.

`--stub` is the loop to develop in. A real chat turn costs tens of seconds and a boot costs
seconds more; the stub answers straight away and deliberately cycles through every branch the UI
has to render — a grounded answer, an answer with *no* evidence, a Reasoned Answer clarification
request, an unresolved clip path, and a failure (ask it anything containing "fail").

## Layout

Follows the project's hexagonal convention, plus a Qt-only `ui/` layer:

```
onyx_desktop/
├── domain/      frozen dataclasses, enums, and IEngineGateway — no Qt, no app.*
├── data/        engine_gateway.py (the ONLY module importing app.*), stub_gateway, session_store
├── service/     engine_bridge / engine_worker (threading), log_relay, desktop_app
├── ui/          theme, main_window, views/, widgets/
├── exceptions/  DesktopError and friends
└── utils/       error_mapper, formatting
```

The dependency rule that makes the stub possible: **`ui/` and `service/` never import `app.*`.**
Only `data/engine_gateway.py` does, and it maps engine entities into the frozen dataclasses in
`domain/entities.py`. That is also why the whole UI is testable without a GPU.

## Views

Chat, Search, Ingest, **Report**. The Report view is the odd one out, deliberately:

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
  button is deliberately disabled with a tooltip saying so. Closing the window mid-ingestion asks
  whether to quit anyway and warns that partially indexed material may be left behind.
- **Ingestion progress is partly blind.** Progress comes from a `JobManager` callback that fires
  once per file per phase, plus the engine's log stream. Eleven batch phases have no logger at all
  — including transcription and diarization, the two longest — so the display can sit still for
  many minutes. The view says so explicitly ("no output for 4m 12s") rather than looking hung.
  The durable fix is two `logger.info` lines per silent phase upstream; the view already
  understands them and would start moving with no change here.
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
