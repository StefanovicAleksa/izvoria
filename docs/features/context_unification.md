# context_unification

`app/features/context_unification/`

## What it does

The final deterministic assembly step that fuses audio-derived textual context and visual-scene context into a single RAG-ready text payload per chunk — the last stop before clip extraction and vector indexing.

## How it does it

`UnificationAPI.execute` takes a `ContextualizedTranscript` (audio context) and an optional `VisualTimeline` (visual context — optional because audio-only sources have no video branch at all). It validates that if a `VisualTimeline` is provided, its `source_id` matches the audio transcript's, raising `TranscriptMismatchError` otherwise — a hard guard against cross-file data corruption.

The actual stitching (`utils/assembler.assemble_unified_chunks`) is pure Python string templating with zero VRAM/model cost: it builds an O(1) lookup dict of visual chunks keyed by `chunk_id`, then for each audio chunk produces either:

```
[Audio Context: ...]
[Visual Context: ...]

<original_text>
```

when a matching non-empty visual chunk exists, or just `[Audio Context: ...]\n\n<original_text>` as a fallback when there's no screen activity for that time window.

That stitched string is `text_payload` — what gets embedded and keyword-indexed. The pieces are also kept apart: `raw_text` is the verbatim speaker-labelled transcript (`original_text`), `context_text` the generated `[Audio Context]`/`[Visual Context]` header, and `speaker_labels` the chunk's speakers in order of first appearance. The answer model only ever reads `raw_text`.

## Inputs / Outputs

Input: `ContextualizedTranscript`, optional `VisualTimeline`. Output: `UnifiedTranscript` (`chunks: List[UnifiedChunk]`, each with `text_payload`, `raw_text`, `context_text`, `speaker_labels` and `visual_evidence: List[VisualContextFrame]`) — this is exactly what `clip_extractor` later consumes as `transcript.chunks`.

## Where it fits

Phase 15 (`ContextUnificationPhase`) — runs after the (conditional) visual branch, immediately before clip extraction and RAG indexing (which run concurrently in `BatchMediaPipeline`).

## Gotchas / design notes

Idempotent via `get_by_source_id`. The source-id mismatch guard is the only real "business logic" here beyond templating — this feature is deliberately a pure, deterministic, zero-cost final-assembly step in an otherwise GPU-heavy pipeline, and should stay that way.
