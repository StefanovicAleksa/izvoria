# frame_extraction

`app/features/frame_extraction/`

## What it does

Pulls representative still-frame images (JPEGs) from video for each RAG chunk flagged as needing visual evidence, filtering out blurry and visually-duplicate frames, to provide visual evidence tied to the transcript.

## How it does it

`FrameExtractionAPI.process_chunks_batch` runs chunks on a `ThreadPoolExecutor` of `FRAME_EXTRACTION_WORKERS` workers (0 = `max(2, cpu_count // 2)`). Each worker drives one FFmpeg decode with `FRAME_FFMPEG_THREADS` (2) threads, and FFmpeg subprocesses and OpenCV release the GIL. Per chunk (`process_chunk`, idempotent via `get_by_chunk_id`), a 4-step pipeline runs:

1. `FFmpegExtractionAdapter.extract_raw_frames` decodes the chunk's interval **once**: `-ss start -i video -t duration -vf fps=1 -q:v 2 -threads 2`, with resilience flags (`-err_detect ignore_err -fflags +discardcorrupt`) for corrupted source video, writing `frame_00000.jpg`, `frame_00001.jpg`, … Frame *i* is stamped `start_sec + i / fps`; frames at or past `end_sec` are dropped. Frames written before an FFmpeg error are kept. If nothing is produced (e.g. a sub-second interval), one frame is taken at the interval midpoint via `extract_frame`. The timeout is `max(60s, 4 × duration)`.
2. `QualityFilter.filter_frames` computes OpenCV Laplacian variance (sharpness) per frame via `cv2.Laplacian(gray, cv2.CV_64F).var()` and deletes frames below `FRAME_BLUR_THRESHOLD` (100). If **no** frame passes, the single sharpest frame is kept anyway (unless it is perfectly uniform, score 0): low-texture evidence such as a scanned page or a mostly white slide scores low while still being what the chunk refers to.
3. `FrameDeduplicator.deduplicate` computes a 64-bit perceptual hash (`imagehash.phash`) per surviving frame, comparing sequentially against the previous unique frame's hash via Hamming distance (`hamming_threshold=5`) to collapse static/locked-off shots.
4. `utils/frame_selector.select_diverse_frames` caps the chunk at `FRAME_MAX_PER_CHUNK` (4) frames by greedy farthest-point selection on pHash distance: it starts from the sharpest frame and repeatedly adds the frame farthest from everything already chosen (ties go to the sharper frame). The result is returned in chronological order.

Survivors are vaulted via `StorageAPI.ingest_raw_file` and mapped into `ExtractedFrame` records; unselected frames are removed with the temporary directory.

## Inputs / Outputs

Input: `source_id`, `video_path`, `List[ContextualizedChunk]`. Output: `List[ExtractedChunkFrames]`, each containing `frames: List[ExtractedFrame]` (`file_id`, `timestamp_sec`, `phash_value`, `quality_score`).

## Where it fits

Phase 11 (`FrameExtractionPhase`) — runs after visual cue detection (for the flagged chunks only), before VLM analysis.

## Gotchas / design notes

- If a single thread in the batch raises, the exception propagates and fails the whole batch/job — there's no per-chunk isolation at the batch level.
- Chunks where no frames survive filtering are still explicitly persisted as empty `ExtractedChunkFrames` (not omitted), to preserve idempotency semantics on re-runs.
- Frame timestamps are nominal (`start + i / fps`): the fps filter emits the decoded frame nearest each tick.
- VLM cost scales with frame count, so `FRAME_MAX_PER_CHUNK` is the main throughput lever for the visual branch. Before the cap (and the single-pass decode), a long flagged chunk sent every changed second to the VLM, one FFmpeg process per second of footage.
