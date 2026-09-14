# jobs

`app/core/jobs/`

## What it does

A generic job-lifecycle/audit-trail wrapper used by every orchestration `Phase` to record what ran, when, with what result, independent of what the wrapped feature actually does. This is how the system gets a consistent history of every pipeline stage execution — pending → processing → completed/failed — regardless of which feature it's wrapping.

## How it does it

`JobManager.run_job(request: JobRequest, target_func, *args, **kwargs)` (`service/manager.py`) is the sole entry point:

1. Creates the job row via `IJobRepository.create_job`.
2. Calls `start_job` (must succeed or raises `JobExecutionFailed`).
3. Invokes `target_func(*args, **kwargs)` inside a try/except.
4. On success: a compact JSON-safe summary is built — dict results keep their scalar entries (keys starting with `_` carry transient in-memory payloads and are dropped); any other result is reduced to `{result_type, result_id}` so whole transcripts never land in the jobs table — then `finish_job` stores it in the `output_params` column.
5. On any exception: captures `traceback.format_exc()`, calls `fail_job(error_message, stack_trace)`, and **re-raises the original exception** — so the calling Phase/Pipeline still sees the real typed error (e.g. `ScannerError`), the job row is just a side-effect record of it.

`JobRequest` (frozen entity) holds `job_type: JobType`, optional `source_id`, and `input_params`. `Job` adds `status`, `worker_id`, `output_result` (read from the `output_params` column), error fields, timing fields, and a `computed_field duration_ms`. `JobType` enumerates every pipeline phase (directory_scan through qa_generation) — it effectively documents the full phase list in one enum. `JobStatus` is `PENDING`/`PROCESSING`/`COMPLETED`/`FAILED`/`CANCELLED`. `JobManager` also exposes `cancel_job()` and `get_source_history()`.

## Inputs / Outputs

Input: a `JobRequest` plus the callable to run. Output: a `Job` record (persisted) and, on the calling side, whatever `target_func` returned (or the re-raised original exception on failure).

## Where it fits

Wraps every orchestration `Phase` (see `../orchestration.md`) — each Phase calls `job_manager.run_job(...)` around the feature `service` API call it wraps, so job tracking is uniform across all ~24 phases.

## Gotchas / design notes

- No retry logic exists here — one failure marks the job `FAILED` and the original exception bubbles straight up to the Phase/Pipeline, which decides whether to halt the batch or continue with the next item.
- Because the original exception is always re-raised, job tracking is purely additive — removing `JobManager` from a call path would not change error-handling behavior, only remove the audit trail.
