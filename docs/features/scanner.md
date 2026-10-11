# scanner

`app/features/scanner/`

## What it does

Recursively walks a filesystem directory to discover ingestible media/document files. Since Izvoria is on-prem/offline, this is how new client files (dropped into a watched folder) get discovered in the first place.

## How it does it

`FileScanner.scan` (implements `IScanner`) uses `os.walk` with a custom `onerror` callback that **re-raises** `OSError` — by default `os.walk` silently skips unreadable directories, and this is explicitly patched so directory-level access problems aren't hidden. Directories are filtered in-place against `IGNORED_DIRS` (`.git`, `__pycache__`, `venv`, `node_modules`, etc.) and files against `ALLOWED_EXTENSIONS`, a union of `VIDEO_EXTENSIONS`, `AUDIO_EXTENSIONS`, `TEXT_EXTENSIONS` and `IMAGE_EXTENSIONS` (defined in `utils/filters.py`). The video ingest filter (`IngestPhase`) and the audio pipeline import these lists rather than keeping their own. The audio pipeline used to accept `.ogg`/`.opus`/`.wma`/… that the scan had already dropped. It's implemented as a generator, yielding one `ScannedFile` per matching file. Individual file `stat()` failures are caught and skipped (file-level resilience), while directory-level errors from `os.walk` are surfaced and mapped via `map_scanner_error`.

## Inputs / Outputs

Input: `root_path: str | Path`. Output: `Generator[ScannedFile, None, None]` where `ScannedFile` (frozen/immutable pydantic model) = `path`, `file_name`, `extension`, `size_bytes`.

## Where it fits

Phase 1 (`ScanPhase`) of every batch pipeline (media, audio, document, image) — the very first step of ingestion, before anything is registered as a `Source`. The scan is shared; each pipeline then keeps only its own suffixes, and the four lists are disjoint, so a folder mixing recordings, documents and images is split between the modes without overlap.

## Gotchas / design notes

**Izvoria's own output is never scanned** (`is_onyx_output`). The vault stores frames, clips, extracted audio and copies of every ingested file under `<storage root>/YYYY/MM/DD` (`ONYX_STORAGE_PATH`, default `onyx_data`), and finished reports under `REPORTS_DIR`. A scan that reached them would re-ingest video frames as images, clips as videos and — worst — the model-written reports (`.md`) as source documents, citing Izvoria's own prose as evidence. Those directories are pruned from the walk, and a scan rooted inside one yields nothing. Everything else under the storage root (the language benchmark's `benchmarks/<code>/media`) is still scanned.

Deliberate two-tier error strategy: directory access errors abort the whole scan, but individual unreadable files are silently skipped — so a scan can return a partial file list without any warning surfaced to the caller.
