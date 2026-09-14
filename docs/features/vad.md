# vad

`app/features/vad/`

## What it does

Voice Activity Detection — scans an extracted audio file and produces a map of exactly which time ranges contain human speech, used to skip silence for transcription, diarization, and domain adaptation. Internally described as "the Rosetta Stone" — nearly every later audio-side feature reads this map.

## How it does it

`VadAPI.process_audio` is idempotent, then uses `ModelManagerOrchestrator.acquire_model`/`release_model` for strict VRAM lifecycle management around inference, delegating to `IVadModelAdapter`. The concrete `NeMoVadAdapter` uses NVIDIA NeMo's MarbleNet (`EncDecClassificationModel`, restored locally from a `.nemo` file — no network calls). It resamples/mono-mixes audio to 16kHz via torchaudio, then runs a manual sliding-window sequential inference loop (`WINDOW_LENGTH_SEC=0.63`, `SHIFT_LENGTH_SEC=0.08`), softmaxing the class-1 ("Speech") probability per window. Speech blocks are built via hysteresis smoothing: `ONSET_THRESHOLD=0.50` to start a block, `OFFSET_THRESHOLD=0.40` to end it, discarding blips under `MIN_SPEECH_DURATION=0.25s`. Confidence per segment is the mean of accumulated frame probabilities.

## Inputs / Outputs

Input: `source_id`, `audio_path`. Output: `VadMap` (persisted as JSONB) containing `List[VadSegment]` (`start_sec`, `end_sec`, `confidence`), with a `total_speech_seconds` convenience property.

## Where it fits

Phase 4 (`VadPhase`) — runs right after audio extraction, before domain adaptation and transcription, both of which consume its output.

## Gotchas / design notes

- Deliberately runs a frame-by-frame Python loop rather than batched inference — explicitly documented as "memory-safe sequential inference" (a VRAM-conservative choice over raw speed).
- Trailing speech at end-of-file is handled specially so a final in-progress segment isn't dropped.
