# visual_context_synthesis

`app/features/visual_context_synthesis/`

## What it does

Turns the accepted (non-hallucinated) per-frame visual descriptions for a chunk into a single fluent 1-2 sentence paragraph — producing the final `VisualTimeline` that gets merged with the audio transcript during RAG indexing.

## How it does it

`VisualSynthesisAPI.execute()` is idempotent (checks `IVisualSynthesisRepository.get_by_source_id` first). For each transcript chunk it looks up the hallucination-correction ledger to find which chunks were accepted; only accepted chunks get visual synthesis work. It fetches the visual-analysis records for those chunks, then calls `build_fuzzy_visual_timeline()` (`utils/fuzzy_grouper.py`), which sorts frame analyses by timestamp and greedily merges consecutive frames into "events" using Jaccard word-set similarity (threshold 0.65, with an NLTK stopword + vision-noise-word filter) — this deduplicates near-identical VLM descriptions (e.g. several frames of the same static slide) into a compact timeline.

Each event's frame list is handed to `IVisualSLMAdapter.synthesize_context()`; the concrete `VisualSLMAdapter` (Llama 3.2 3B GGUF, `n_ctx=4096`, `temperature=0.0`) formats a timestamped bullet list and asks the SLM to synthesize it into a single paragraph. Produces a `VisualTimelineChunk(visual_context_string, visual_evidence=[VisualContextFrame])` per chunk; chunks without any accepted evidence get an empty placeholder (all entities are frozen/immutable).

## Inputs / Outputs

Input: `ContextualizedTranscript` + the hallucination-correction ledger + visual-analysis records. Output: `VisualTimeline` (`chunks: List[VisualTimelineChunk]`).

## Where it fits

Phase 14 (`VisualContextSynthesisPhase`) — the last visual-branch phase, running right before context unification merges its output with the audio-derived transcript.

## Gotchas / design notes

- The NLTK `stopwords` corpus auto-downloads on first import if not already present locally — a network dependency at import time worth knowing about on a genuinely offline/air-gapped deployment (it should be pre-downloaded as part of setup, not fetched at runtime).
- Frame analyses carry their frame's `timestamp_sec` (copied from `ExtractedFrame` by `visual_analysis`), so timeline events get real start/end times. `base_timestamp_sec` (the chunk start) is only a fallback for records written before that field existed. Previously `FrameAnalysis` had no timestamp, so every event was stamped with the chunk start.
