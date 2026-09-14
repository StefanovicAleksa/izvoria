# clip_extractor

`app/features/clip_extractor/`

## What it does

Cuts short physical `.mp4` video clips corresponding to each chunk of a `UnifiedTranscript`, so the UI/RAG system can show the exact video moment behind a piece of retrieved text.

## How it does it

`ClipExtractorAPI.extract_clips_from_transcript` is idempotent per source (`get_by_source_id`), validates the source is `SourceType.VIDEO` with an attached file, then loops over `transcript.chunks`, padding each chunk by ±0.5s buffer. For each it calls `IFFmpegVideoAdapter.extract_clip`. The concrete `FFmpegVideoAdapter` runs `ffmpeg -ss <start> -i <src> -t <end - start> -c copy` — placing `-ss` before `-i` for fast keyframe seeking (input seeking resets timestamps to 0, so the length must be a duration: an output `-to <end>` is also treated as a duration and produced clips far longer than the chunk), and using stream copy (`-c copy`) to avoid CPU re-encoding entirely. Each resulting temp `.mp4` is vaulted via `StorageAPI.ingest_raw_file(delete_original=True)`, wrapped into a `VideoClip` entity, and all clips are bulk-persisted via `IVideoClipRepository.create_bulk`.

## Inputs / Outputs

Input: a `UnifiedTranscript` (from `context_unification`). Output: `List[VideoClip]` (`id`, `source_id`, `chunk_id`, `extracted_file_id`, `start_sec`, `end_sec`).

## Where it fits

Phase 15.5 (`ClipExtractionPhase`) — runs concurrently with RAG indexing (Phase 16) in `BatchMediaPipeline`, via a shared `ThreadPoolExecutor(max_workers=2)`, since one is pure CPU/ffmpeg work and the other is the sole GPU consumer at that point.

## Gotchas / design notes

- `-c copy` means clip boundaries snap to the nearest keyframe rather than being frame-exact — a clip's actual start/end may be slightly off from the requested `start_sec`/`end_sec`.
- A failure mid-loop cleans up the current temp file, deletes every clip already vaulted in this run (`StorageAPI.delete_raw_file`, since they have no DB rows yet) and re-raises, aborting the whole batch — no partial persistence and no orphaned vault files.
