# diarization

`app/features/diarization/`

## What it does

Determines "who spoke when" (speaker attribution) and aligns those speaker labels onto the existing Whisper transcript.

## How it does it

`DiarizationAPI.process_audio` requires both a `VadMap` and an existing `Transcript` as prerequisites (raises `DiarizationManifestError` otherwise), then runs NVIDIA NeMo's `ClusteringDiarizer` (TitaNet speaker embeddings) via `NeMoDiarizationAdapter`. It forces `gc.collect()`/`torch.cuda.empty_cache()` before running to defragment VRAM after prior GPU-heavy phases.

It builds an "Oracle VAD" RTTM file directly from the existing `VadMap` (skipping NeMo's own internal VAD stage entirely) plus an input manifest JSON, and configures `OmegaConf` with a 2-scale multiscale clustering setup (`window_length_in_sec: [1.5, 0.5]`) as a "VRAM compromise" between accuracy and the O(N²) cost of spectral clustering, capped at `max_num_speakers: 20`. If clustering fails (linear algebra/convergence errors — typically a single-speaker monologue) or the RTTM output is missing, it gracefully falls back to labeling everything `"Speaker_0"` rather than crashing. Output RTTM lines are parsed into `DiarizationSegment`s.

Finally, `utils/aligner.align_transcript_with_diarization` attributes speakers **per word**: each `TranscriptWord` gets the diarization speaker it overlaps most (or the nearest speaker if it falls in a gap; `"Unknown"` only when the map is empty), using a sorted index with a prefix-max of end times. A Whisper segment whose words span a speaker change is split into one segment per speaker run (text rebuilt from its words, Whisper metadata copied); segments without word timestamps fall back to segment-level max overlap. Diarization clustering runs NeMo's long-form mode (`chunk_cluster_count`, `embeddings_per_chunk`) so multi-hour files don't build one giant affinity matrix. and the updated `Transcript` is persisted back via `TranscriptionAPI.update_transcript`. The `DiarizationMap` is saved **last**: it is the idempotency marker, so a crash between alignment and saving simply reruns diarization instead of skipping alignment forever.

## Inputs / Outputs

Input: `source_id`, `audio_path` (needs `VadMap` + `Transcript` as prerequisites). Output: `DiarizationMap` (`segments: List[DiarizationSegment]` with `speaker_id`), plus a side-effect update to the stored `Transcript`.

## Where it fits

Phase 6 (`DiarizePhase`) — runs after hallucination correction, before sentence reconstruction (which needs speaker boundaries to know where not to merge text).

## Gotchas / design notes

- The fallback-to-monologue behavior is a deliberate defensive design decision, logged as a `[DIARIZATION FALLBACK]` warning and recorded as `DiarizationMap.fallback_applied=True` (persisted) — not a silent failure.
- Word-level alignment exists because a single Whisper segment often contains a question and its answer; segment-level attribution gave the answer to the asker.
