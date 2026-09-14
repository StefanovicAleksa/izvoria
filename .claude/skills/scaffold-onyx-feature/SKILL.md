---
name: scaffold-onyx-feature
description: Scaffold a new Onyx feature folder (domain/data/service/exceptions/utils) following the project's hexagonal feature-first convention, and produce the integration checklist for wiring it into config, the bootstrapper, and orchestration. Use when asked to add a new feature, adapter, pipeline stage, or module to app/features, app/core, app/legacy_features, or app/tools.
---

# Scaffold an Onyx feature

Onyx (see `CLAUDE.md` at the repo root) uses one repeated hexagonal folder convention across every feature. This skill generates a new feature's skeleton in that convention and hands back the checklist of manual integration steps that can't be auto-generated. Read `CLAUDE.md`'s "Architecture" and "How to add a new feature" sections first if this session hasn't already.

## Step 1 — Establish shape

Before creating anything, determine (ask the user if it isn't already clear from the request):

1. **Feature name** — snake_case, matching the pattern of existing features (e.g. `retrieval_hallucination_scrubber`, `visual_context_synthesis`).
2. **Where it lives** — `app/features/<name>` for the normal case. Only use `app/core/<name>` for cross-cutting infrastructure (database, jobs, model management), `app/tools/<name>` for a dev-only tool, or `app/legacy_features/` if explicitly told this supersedes/replaces something being deprecated.
3. **Is it GPU/model-bound?** If yes, it needs a `data/` adapter implementing `IModelLoader`, plus a `ModelManifest` entry in `app/core/config.py`. If it's pure logic or DB-only, it may not need `data/` at all — see the no-adapter template below.
4. **Does it persist to Postgres?** If yes, it needs a `data/` repository adapter too, even if it isn't GPU-bound.

## Step 2 — Pick a template from the existing codebase

Don't invent new conventions — copy the shape of an existing feature that matches:

- **No adapter needed** (pure logic/heuristics, no model, no DB): use `app/features/hallucination_correction/` as the template. Its shape: `domain/{entities.py,interfaces.py}`, `exceptions/{base.py,<specific>.py}`, `service/<name>_api.py`, `utils/{error_mapper.py,<helpers>.py}` — no `data/`.
- **Model adapter + DB repository** (loads a model and/or persists results): use `app/features/rag_retrieval/` as the template. Its shape adds `data/{<model>_adapter.py or <model>_repo.py}` implementing the interfaces declared in `domain/interfaces.py`, plus a richer `exceptions/` set (`base.py`, `database.py`, `model.py`, `storage.py` as needed).

Read the template feature's actual files before writing the new ones — match its style (docstrings on interfaces describing the contract in one sentence, typed exceptions per failure category, `service/*_api.py` as the sole public surface) rather than guessing.

## Step 3 — Create the skeleton

For a feature at `app/features/<name>/`, create:

```
app/features/<name>/__init__.py
app/features/<name>/domain/__init__.py
app/features/<name>/domain/entities.py       # dataclasses/pydantic models for this feature's domain objects
app/features/<name>/domain/interfaces.py      # ABCs — the ports (e.g. I<Name>Adapter, I<Name>Repository)
app/features/<name>/domain/enums.py           # only if the feature has meaningful enums; omit otherwise
app/features/<name>/exceptions/__init__.py
app/features/<name>/exceptions/base.py        # <Name>Error base class
app/features/<name>/service/__init__.py
app/features/<name>/service/<name>_api.py     # the public entrypoint — this is what Phases/other features call
app/features/<name>/utils/__init__.py
app/features/<name>/utils/error_mapper.py     # translates low-level exceptions to this feature's typed exceptions
```

Add, only if the feature needs it:

```
app/features/<name>/data/__init__.py
app/features/<name>/data/<model_or_repo>_adapter.py    # implements an interface from domain/interfaces.py
```

Also create the mirrored unit test directory (even if stubbed with a single `pass`-level test to start):

```
tests/features/<name>/service/test_<name>_api.py
tests/features/<name>/utils/test_error_mapper.py
tests/features/<name>/data/test_<adapter>.py   # if data/ exists
```

Seed each file with real content matching the chosen template's actual patterns (interface docstrings stating the contract in one sentence, e.g. `"""Contract for persisting X to PostgreSQL."""`; exceptions as a small typed hierarchy; `error_mapper.py` as a `@staticmethod`/function table mapping specific low-level exception types to the feature's typed ones) — don't leave placeholder TODOs in files that are supposed to compile and import cleanly.

## Step 4 — Hand back the integration checklist

Auto-generating the skeleton does **not** wire the feature in. After creating files, explicitly list these remaining steps for the user (and do them if asked to complete the integration, not just scaffold it):

- [ ] If GPU/model-bound: add `<FEATURE>_MODEL_PATH` and `<FEATURE>_VRAM_REQUIREMENT_MB` to `Settings` in `app/core/config.py`, following the existing naming pattern.
- [ ] Register the feature's adapter(s) and API in `OnyxBootstrapper.build()` (`app/engine/service/bootstrap.py`).
- [ ] If it belongs in a pipeline: add a `Phase` class in `app/orchestration/service/phases/` (wrapping the new `service/*_api.py`, with preload/unload calls around any model use) and register it in the relevant `Pipeline` in `app/orchestration/service/pipelines/`.
- [ ] If it persists data: confirm the Postgres schema/table exists (via `scripts/init.sql` or a migration) before integration tests can pass.
- [ ] Add an integration test under `integration_tests/features/<name>/` if the feature touches the DB or loads a real model — follow the self-skip pattern (`pytest.skip(...)` when hardware/weights aren't present) used elsewhere in `integration_tests/`.
- [ ] Flesh out the stubbed unit tests created in Step 3 with real mocked-dependency coverage (see `CLAUDE.md`'s Testing section for the mocking conventions used in `tests/`).

## Notes

- Never let a new feature import another feature's `data/` module directly — only import through the other feature's `service/*_api.py`.
- If the feature loads a model, the `data/` adapter must implement `IModelLoader` and all acquisition must go through `ModelManagerOrchestrator` (see `CLAUDE.md`'s "Model management rules") — don't have the adapter load the model directly in `__init__`. Follow the `onyx-model-lifecycle` skill for the adapter, API, Phase and pipeline patterns.
