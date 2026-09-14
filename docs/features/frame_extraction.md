# frame_extraction

`app/features/frame_extraction/`

## What it does

Pulls representative still-frame images (JPEGs) from video for each RAG chunk flagged as needing visual evidence, filtering out blurry and visually-duplicate frames, to provide visual evidence tied to the transcript.

## How it does it

`FrameExtractionAPI.process_chunks_batch` parallelizes chunk processing across a `ThreadPoolExecutor` (capped at `min(32, cpu_count*2)` workers — safe because the actual work is FFmpeg subprocesses and OpenCV C++ bindings, both of which release the GIL). Per chunk (`process_chunk`, idempotent via `get_by_chunk_id`), a 3-step pipeline runs:

1. `FFmpegExtractionAdapter.extract_raw_frames` seeks and extracts one JPEG per second (`fps=1`, `-q:v 2`) strictly within the chunk's `[start_sec, end_sec)`, using resilience flags (`-err_detect ignore_err -fflags +discardcorrupt`) for corrupted source video, with a 30s timeout per frame.
2. `QualityFilter.filter_frames` computes OpenCV Laplacian variance (sharpness) per frame via `cv2.Laplacian(gray, cv2.CV_64F).var()`, discarding (and deleting from disk) blurry frames below `blur_threshold=100.0`.
3. `FrameDeduplicator.deduplicate` computes a 64-bit perceptual hash (`imagehash.phash`) per surviving frame, comparing sequentially against the previous unique frame's hash via Hamming distance (`hamming_threshold=5`) to collapse static/locked-off shots.

Survivors are vaulted via `StorageAPI.ingest_raw_file` and mapped into `ExtractedFrame` records.

## Inputs / Outputs

Input: `source_id`, `video_path`, `List[ContextualizedChunk]`. Output: `List[ExtractedChunkFrames]`, each containing `frames: List[ExtractedFrame]` (`file_id`, `timestamp_sec`, `phash_value`, `quality_score`).

## Where it fits

Phase 11 (`FrameExtractionPhase`) — runs after visual cue detection (for the flagged chunks only), before VLM analysis.

## Gotchas / design notes

- If a single thread in the batch raises, the exception propagates and fails the whole batch/job — there's no per-chunk isolation at the batch level.
- Chunks where no frames survive filtering are still explicitly persisted as empty `ExtractedChunkFrames` (not omitted), to preserve idempotency semantics on re-runs.
