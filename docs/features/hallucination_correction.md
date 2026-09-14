# hallucination_correction

`app/features/hallucination_correction/`

## What it does

A deterministic, zero-GPU post-processing filter that removes Whisper hallucinations (fabricated text) from a transcript — a well-known Whisper failure mode on silence, music, or background noise.

## How it does it

`CorrectionAPI.process_transcript` requires an existing `Transcript` and `VadMap`, then runs every segment through `CorrectionHeuristicsEngine.evaluate_segment`, a pure math/regex "3-step gauntlet" with **no model inference** (explicitly "Zero-VRAM"):

1. **VAD strict intersection** — compute overlap between the transcript segment and all VAD speech segments; if overlap ratio < `min_vad_overlap_ratio` (0.15 default), flag as hallucination.
2. **Whisper metadata** — delete only when low confidence (`avg_logprob < -1.0`) coincides with likely silence (`no_speech_prob > 0.60`) or a repetition loop (`compression_ratio > 2.4`), mirroring Whisper's own silence rule (thresholds from `CorrectionConfig`). Any one of those signals alone does **not** delete the segment: accented, jargon-heavy or poorly recorded real speech often trips one of them. Such segments are kept and marked `TranscriptSegment.low_confidence=True` (`CorrectionHeuristicsEngine.is_low_confidence`).
3. **Regex blacklist** against known hallucination phrases (e.g. "subscribe", "amara.org", broadcast network names, `[Music]`/`[Applause]` tags).

Surviving segments replace the transcript's segment list; a `CorrectionTelemetry` counts removals per category and `flagged_low_confidence` for debugging.

## Inputs / Outputs

Input: `source_id` (loads `Transcript` + `VadMap` internally). Output: `Tuple[Transcript, CorrectionTelemetry]`.

## Where it fits

Phase 5.5 (`HallucinationCorrectionPhase`) — runs immediately after transcription, before diarization, so speaker alignment never has to deal with fabricated segments.

## Gotchas / design notes

- Raises `CorruptSegmentDataError` if a segment has zero/negative duration, rather than silently skipping it.
- Relies entirely on Whisper's own confidence metadata (`no_speech_prob`, etc.), which must be populated upstream by the transcription adapter — this feature has no fallback if that metadata is missing.
