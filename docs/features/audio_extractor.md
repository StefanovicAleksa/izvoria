# audio_extractor

`app/features/audio_extractor/`

## What it does

Extracts a standalone audio track from an ingested video `Source` so downstream audio-only features (VAD, transcription) have something to work with.

## How it does it

`AudioExtractorAPI` (`service/extractor_api.py`) is idempotent — it first checks `IAudioExtractionRepository.get_latest_by_source_id` and returns the cached record if present. Otherwise it fetches the `Source` (via `storage`), validates it's `SourceType.VIDEO` with an attached `File`, then calls `IFFmpegAdapter.extract_audio`. The concrete `FFmpegAdapter` shells out to `ffmpeg` via `subprocess.run` to produce a 16kHz mono 16-bit PCM WAV (`pcm_s16le`) into a `tempfile.mkstemp` path. If ffmpeg reports the video has no audio stream at all ("does not contain any stream"/"Invalid argument"), it falls back to generating a synthetic 1-second silent WAV via Python's `wave`/`struct` modules, so a video with no audio track doesn't break the pipeline. The resulting temp file is vaulted via `StorageAPI.ingest_raw_file(delete_original=True)`, and an `AudioExtraction` lineage record is persisted.

## Inputs / Outputs

Input: `video_source_id: uuid.UUID`. Output: `AudioExtraction` entity (`id`, `source_id`, `extracted_file_id`).

## Where it fits

Phase 3 (`ExtractAudioPhase`) — runs right after `IngestPhase`, before VAD.

## Gotchas / design notes

- Temp file cleanup on failure is manual (try/except with `unlink`), not automatic.
- The silent-track fallback prioritizes pipeline integrity over surfacing a genuinely broken video file — worth knowing when debugging an unexpectedly empty transcript.
