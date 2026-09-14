# transcription

`app/features/transcription/`

## What it does

Converts speech audio into text using OpenAI Whisper (via faster-whisper), producing word- and segment-level timestamped transcripts — the primary text payload that eventually feeds the RAG index.

## How it does it

`TranscriptionAPI.process_audio` is idempotent, fetches the precomputed `VadMap` (returns an empty `Transcript` if there's no speech at all — it never transcribes silence), acquires the Whisper model via the VRAM orchestrator, and delegates to `ITranscriptionAdapter.transcribe`. `FasterWhisperAdapter` loads a `faster_whisper.WhisperModel` with `device="cuda", compute_type="float16"`.

It reads the extracted 16kHz mono WAV and only ever transcribes VAD speech — this is how transcription "bypasses" silent sections rather than wasting compute on them.

- **Batched path** (`WHISPER_BATCH_SIZE` > 1, default 8): `_plan_batched_clips` packs VAD segments into clips of at most 29.5s that start and end only at VAD boundaries (i.e. in silence, joining segments across gaps ≤2s), and faster-whisper's `BatchedInferencePipeline` decodes them in parallel via `clip_timestamps`. Batched mode only transcribes the first 30s of a clip, so any single VAD segment longer than that is routed to the sequential path instead of being truncated. Batched timestamps are already absolute.
- **Sequential path** (long uninterrupted speech, or `WHISPER_BATCH_SIZE=1`): merges adjacent VAD segments within a 2-second gap (`_merge_vad_segments`) and transcribes each span with `WhisperModel.transcribe`, offsetting timestamps by the span start.
- Spans shorter than `WHISPER_MIN_SLICE_SEC` (1s) are widened symmetrically within the audio instead of being skipped, so isolated one-word answers ("Yes.") are not lost. Both paths pass `hotwords`/`initial_prompt` (truncated to 45 words) sourced from `domain_adaptation`, use `word_timestamps=True`, `condition_on_previous_text=False`, and `vad_filter=False` (VAD is already handled upstream). Results are sorted by start time.

## Inputs / Outputs

Input: `source_id`, `audio_path`, optional `language`/`hotwords`/`initial_prompt`. Output: `Transcript` (`segments: List[TranscriptSegment]`, each with `words: List[TranscriptWord]`, plus Whisper metadata `no_speech_prob`/`avg_logprob`/`compression_ratio` used later by `hallucination_correction`). `Transcript.full_text` joins all segment text.

## Where it fits

Phase 5 (`TranscribePhase`) — runs after domain adaptation (which conditions it) and VAD (which windows it), before hallucination correction and diarization.

## Gotchas / design notes

- `speaker_id` defaults to `"Unknown"` at this stage — `diarization` fills it in later.
- `condition_on_previous_text=False` is a deliberate choice to prevent hallucination cascades from propagating across independently-sliced VAD chunks.
