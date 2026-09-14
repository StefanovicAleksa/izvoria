# storage

`app/features/storage/`

## What it does

The central "vault" feature — the single place responsible for physically storing files on disk and tracking `Source`/`File` records in Postgres, including content-hash based deduplication. Every other feature that needs to persist a derived artifact (extracted audio, a frame, a clip) or register a newly discovered source goes through this feature.

## How it does it

`StorageAPI` wraps `IStorageRepository` (DB) and a concrete `FileManager` (data layer, not behind an interface). `FileManager.generate_path` builds a date-sharded path (`onyx_data/YYYY/MM/DD/<uuid><ext>`); `store_file` either `shutil.move`s or `shutil.copy2`s the source in, then `os.chmod(dest_path, 0o644)`. `ingest_local_file` computes a content hash via `utils/hashing.calculate_file_hash`, checks `get_by_hash` for deduplication (deletes the incoming original if a duplicate is found and `delete_original=True`), and otherwise creates both `File` and `Source` records. `ingest_raw_file` is the simpler no-dedup path used by extractor/frame features to just vault a derived artifact; `delete_raw_file` reverses it (DB record + physical file) when a feature has to roll back artifacts it vaulted during a failed run. `get_file_path` does a strict double-check that the DB record's physical file still exists on disk, raising `FileNotFound` otherwise.

## Inputs / Outputs

Inputs: local file paths + a `SourceType` enum (`VIDEO`/`AUDIO`/`TEXT`/`DOCUMENT`). Outputs: `File` and `Source` domain entities.

## Where it fits

Phase 2 (`IngestPhase`) registers newly scanned files as `Source` rows via this feature; nearly every later phase (audio extraction, frame extraction, clip extraction) calls back into `StorageAPI.ingest_raw_file`/`get_file_path` to persist or retrieve derived artifacts.

## Gotchas / design notes

- `store_file`'s `chmod 644` makes vaulted files effectively read-only-by-convention, which could cause a later overwrite attempt elsewhere in the system to fail unexpectedly.
- `FileManager.delete` has a security check to only delete inside `base_path`, but fails silently on `OSError` rather than raising.
