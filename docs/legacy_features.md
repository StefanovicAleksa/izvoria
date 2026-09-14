# Legacy features

`app/legacy_features/`

These five features are kept in the codebase for reference but are **confirmed dead code** — none are wired into `OnyxBootstrapper.build()`, and none run in any current pipeline. Two of them (`hallucination_scrubber`, `intelligence_bridge`) still have orchestration `Phase` classes and `PipelinePhase` enum entries sitting unreferenced ("zombie" scaffolding); the other three have no orchestration presence left at all. Don't build on these without first confirming with the team whether they're meant to be revived or removed.

## asr_error_correction

**What it did:** A 3-phase GPU pipeline for domain-specific ASR error correction, run after sentence reconstruction. Phase 1: Llama 3.2 3B (GGUF) scans the full document to build a `DocumentGlossary` of domain terminology. Phase 2: a T5-ASR-EC model flags acoustic/phonetic anomalies in reconstructed sentences. Phase 3: Qwen 2.5 7B (GGUF) resolves flagged anomalies against the glossary to produce a corrected transcript with an audit trail. Each phase loaded/unloaded its own model independently for VRAM discipline.

**Status:** Fully superseded by [`domain_phonetic_correction`](features/domain_phonetic_correction.md) — same pipeline slot, same glossary-driven correction idea, consolidated to a single adapter with sliding-window chunking instead of a 3-model relay. Zero references outside its own directory.

## hallucination_scrubber

**What it did:** Used a DeBERTa zero-shot classifier to detect and strip semantic hallucinations (YouTube outros, subtitle watermarks, sign-offs, etc.) from finished transcripts without regex — built dynamic candidate labels from the source's domain glossary, did a cheap segment-level "fast path" suspicion check, then a slower clause-by-clause classify/delete pass.

**Status:** Superseded in *concept* by [`retrieval_hallucination_scrubber`](features/retrieval_hallucination_scrubber.md), but not a clean 1:1 replacement — the new one scrubs RAG-retrieved chunks against the user's query at query time, whereas this one scrubbed whole transcripts at ingestion time. **Zombie code**: `orchestration/service/phases/hallucination_scrubber_phase.py` (`HallucinationScrubberPhase`) still exists and imports the legacy API, and `PipelinePhase.HALLUCINATION_SCRUBBER` remains in the enum (commented "Legacy Phase 6.5"), but it is never instantiated in `bootstrap.py` and never referenced by any pipeline — present but unreachable.

## intelligence_bridge

**What it did:** Orchestrated a "two-pass" multimodal timeline builder bridging computer-vision scene detection with semantic NLP: run `scene_detector` globally at a high threshold to catch hard cuts, feed the transcript to `nlp_router` (Phi-3.5) to find semantic "look at this" time windows, re-run scene detection at a low threshold constrained to those windows, then merge/deduplicate both scene maps by perceptual hash.

**Status:** Superseded by the split pipeline of [`frame_extraction`](features/frame_extraction.md) + [`visual_cue_detection`](features/visual_cue_detection.md) + [`visual_analysis`](features/visual_analysis.md) + [`visual_context_synthesis`](features/visual_context_synthesis.md), which decompose "find interesting frames" and "understand them semantically" into separate phases. **Zombie code** like `hallucination_scrubber`: `orchestration/service/phases/intelligence_bridge_phase.py` and `PipelinePhase.INTELLIGENCE_BRIDGE` (commented "Kept in domain, benched from active pipeline") still exist but are never constructed in `bootstrap.py` or called by any pipeline.

## nlp_router

**What it did:** A single-purpose LLM component (Phi-3.5 via llama.cpp) that read chunked transcript text (180s windows) and extracted "visual cue" time windows worth capturing frames from. Had no standalone caller — its only consumer was `intelligence_bridge`.

**Status:** Abandoned along with its sole consumer, `intelligence_bridge`. Its role — deciding which video moments are visually worth extracting — is now handled by [`visual_cue_detection`](features/visual_cue_detection.md) working alongside [`frame_extraction`](features/frame_extraction.md). Referenced only from within `legacy_features` itself.

## scene_detector

**What it did:** Wrapped FFmpeg to detect scene changes via pixel-variance thresholding, extract JPEG frames (globally or constrained to specific time windows), perceptually hash and deduplicate them, then vault surviving frames via `StorageAPI`. Supported both a "global brute-force" mode and a "targeted" mode used exclusively by `intelligence_bridge`.

**Status:** Fully superseded by [`frame_extraction`](features/frame_extraction.md), which has an equivalent FFmpeg adapter plus a more elaborate dedup/quality-filter pipeline and proper DB-backed persistence (this legacy version was pure in-memory, with no repository at all). Referenced only from within `legacy_features` (by `intelligence_bridge` and `nlp_router`).
