# diarization

`app/features/diarization/`

## What it does

Determines "who spoke when" (speaker attribution) and aligns those speaker labels onto the existing Whisper transcript.

## How it does it

`DiarizationAPI.process_audio` requires both a `VadMap` and an existing `Transcript` as prerequisites (raises `DiarizationManifestError` otherwise), then runs NVIDIA NeMo's `ClusteringDiarizer` (TitaNet speaker embeddings) via `NeMoDiarizationAdapter`. It forces `gc.collect()`/`torch.cuda.empty_cache()` before running to defragment VRAM after prior GPU-heavy phases.

It builds an "Oracle VAD" RTTM file directly from the existing `VadMap` (skipping NeMo's own internal VAD stage entirely) plus an input manifest JSON, and configures `OmegaConf` with a 2-scale multiscale clustering setup (`window_length_in_sec: [1.5, 0.5]`) as a "VRAM compromise" between accuracy and the O(N²) cost of spectral clustering, capped at `max_num_speakers: 20`. If clustering fails (linear algebra/convergence errors — typically a single-speaker monologue) or the RTTM output is missing, it gracefully falls back to labeling everything `"Speaker_0"` rather than crashing. Output RTTM lines are parsed into `DiarizationSegment`s.

Finally, `utils/aligner.align_transcript_with_diarization` does a pure-Python max-time-overlap match between each `TranscriptSegment` and all `DiarizationSegment`s, assigning `speaker_id` (default `"Unknown"` if there's no overlap), and the updated `Transcript` is persisted back via `TranscriptionAPI.update_transcript`.

## Inputs / Outputs

Input: `source_id`, `audio_path` (needs `VadMap` + `Transcript` as prerequisites). Output: `DiarizationMap` (`segments: List[DiarizationSegment]` with `speaker_id`), plus a side-effect update to the stored `Transcript`.

## Where it fits

Phase 6 (`DiarizePhase`) — runs after hallucination correction, before sentence reconstruction (which needs speaker boundaries to know where not to merge text).

## Gotchas / design notes

- The fallback-to-monologue behavior is a deliberate defensive design decision, explicitly logged as `[DIARIZATION FALLBACK]` — not a silent failure.
- The transcript/diarization aligner is a brute-force O(segments × diarization_segments) overlap match, not optimized — fine at transcript scale, but worth knowing if diarization becomes a bottleneck.
