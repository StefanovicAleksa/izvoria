# audio_extractor

`app/features/audio_extractor/`

## What it does

Normalizes every audio-bearing `Source` (VIDEO or AUDIO) to one vaulted 16kHz mono PCM WAV, which every downstream audio phase — VAD, domain adaptation, transcription, diarization — reads via `AudioExtractorAPI.get_extracted_audio_path(source_id)`. Reading the original media in each phase meant repeated full decodes/resamples and a different decoder path than the VAD map was computed from.

## How it does it

`AudioExtractorAPI` (`service/extractor_api.py`) is idempotent — it first checks `IAudioExtractionRepository.get_latest_by_source_id` and returns the cached record if present. Otherwise it fetches the `Source` (via `storage`), validates it's `SourceType.VIDEO` or `SourceType.AUDIO` with an attached `File`, then calls `IFFmpegAdapter.extract_audio`. The concrete `FFmpegAdapter` shells out to `ffmpeg` via `subprocess.run` to produce a 16kHz mono 16-bit PCM WAV (`pcm_s16le`) into a `tempfile.mkstemp` path. Before extracting, it counts audio streams with `ffprobe`; only if the file genuinely has none does it fall back to generating a synthetic 1-second silent WAV via Python's `wave`/`struct` modules, so a video with no audio track doesn't break the pipeline. The resulting temp file is vaulted via `StorageAPI.ingest_raw_file(delete_original=True)`, and an `AudioExtraction` lineage record is persisted.

## Inputs / Outputs

Input: `video_source_id: uuid.UUID`. Output: `AudioExtraction` entity (`id`, `source_id`, `extracted_file_id`).

## Where it fits

Phase 3 (`ExtractAudioPhase`) — runs right after `IngestPhase`, before VAD, in both `BatchMediaPipeline` and `BatchAudioPipeline`.

## Gotchas / design notes

- Temp file cleanup on failure is manual (try/except with `unlink`), not automatic.
- The silent-track fallback applies only to media `ffprobe` reports as having zero audio streams; every other FFmpeg failure (corrupt file, stream errors on media that does have audio) raises a typed error instead of silently producing an empty transcript.
