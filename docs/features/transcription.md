# transcription

`app/features/transcription/`

## What it does

Converts speech audio into text using OpenAI Whisper (via faster-whisper), producing word- and segment-level timestamped transcripts — the primary text payload that eventually feeds the RAG index.

## How it does it

`TranscriptionAPI.process_audio` is idempotent, fetches the precomputed `VadMap` (returns an empty `Transcript` if there's no speech at all — it never transcribes silence), acquires the Whisper model via the VRAM orchestrator, and delegates to `ITranscriptionAdapter.transcribe`. `FasterWhisperAdapter` loads a `faster_whisper.WhisperModel` with `device="cuda", compute_type="float16"`.

Before transcribing, it merges adjacent VAD segments within a 2-second gap (`_merge_vad_segments`) to reduce fragmentation and redundant model calls, skips merged chunks under 6400 samples (0.4s), and slices the audio numpy array directly per merged VAD window rather than transcribing the whole file — this is how transcription "bypasses" silent sections rather than wasting compute on them. It passes `hotwords`/`initial_prompt` (truncated to 45 words) sourced from `domain_adaptation`, uses `word_timestamps=True`, `condition_on_previous_text=False`, and `vad_filter=False` (VAD is already handled upstream). Word/segment timestamps are offset by the source VAD segment's `start_sec` to convert relative-to-chunk timing into absolute-to-file timing.

## Inputs / Outputs

Input: `source_id`, `audio_path`, optional `language`/`hotwords`/`initial_prompt`. Output: `Transcript` (`segments: List[TranscriptSegment]`, each with `words: List[TranscriptWord]`, plus Whisper metadata `no_speech_prob`/`avg_logprob`/`compression_ratio` used later by `hallucination_correction`). `Transcript.full_text` joins all segment text.

## Where it fits

Phase 5 (`TranscribePhase`) — runs after domain adaptation (which conditions it) and VAD (which windows it), before hallucination correction and diarization.

## Gotchas / design notes

- `speaker_id` defaults to `"Unknown"` at this stage — `diarization` fills it in later.
- `condition_on_previous_text=False` is a deliberate choice to prevent hallucination cascades from propagating across independently-sliced VAD chunks.
