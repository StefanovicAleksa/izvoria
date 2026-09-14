# scanner

`app/features/scanner/`

## What it does

Recursively walks a filesystem directory to discover ingestible media/document files. Since Onyx is on-prem/offline, this is how new client files (dropped into a watched folder) get discovered in the first place.

## How it does it

`FileScanner.scan` (implements `IScanner`) uses `os.walk` with a custom `onerror` callback that **re-raises** `OSError` — by default `os.walk` silently skips unreadable directories, and this is explicitly patched so directory-level access problems aren't hidden. Directories are filtered in-place against `IGNORED_DIRS` (`.git`, `__pycache__`, `venv`, `node_modules`, etc.) and files against `ALLOWED_EXTENSIONS`, a union of `VIDEO_EXTENSIONS`, `AUDIO_EXTENSIONS`, `TEXT_EXTENSIONS` (defined in `utils/filters.py`). It's implemented as a generator, yielding one `ScannedFile` per matching file. Individual file `stat()` failures are caught and skipped (file-level resilience), while directory-level errors from `os.walk` are surfaced and mapped via `map_scanner_error`.

## Inputs / Outputs

Input: `root_path: str | Path`. Output: `Generator[ScannedFile, None, None]` where `ScannedFile` (frozen/immutable pydantic model) = `path`, `file_name`, `extension`, `size_bytes`.

## Where it fits

Phase 1 (`ScanPhase`) of `BatchMediaPipeline`/`BatchAudioPipeline` — the very first step of ingestion, before anything is registered as a `Source`.

## Gotchas / design notes

Deliberate two-tier error strategy: directory access errors abort the whole scan, but individual unreadable files are silently skipped — so a scan can return a partial file list without any warning surfaced to the caller.
