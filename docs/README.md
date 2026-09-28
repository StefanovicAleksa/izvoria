# Onyx documentation

Per-feature technical documentation — what each module does and how it does it (real class names, algorithms, thresholds, gotchas). For architecture, conventions, and how to add a new feature, see the root [`CLAUDE.md`](../CLAUDE.md). For build history, see the root [`CHANGELOG.md`](../CHANGELOG.md).

## Pipeline order

The batch ingestion pipeline (`BatchMediaPipeline` for video, `BatchAudioPipeline` for audio-only — skips the visual branch) runs these phases in order. The visual branch (10–14) only runs if `visual_cue_detection` flags at least one chunk as needing visual evidence.

1. [scanner](features/scanner.md) — discover files in a watched directory
2. [storage](features/storage.md) — register discovered files as `Source` rows *(ingest)*
3. [audio_extractor](features/audio_extractor.md) — extract audio track from video
4. [vad](features/vad.md) — voice activity detection
4.5. [domain_adaptation](features/domain_adaptation.md) — detect industry/domain, build hotwords
5. [transcription](features/transcription.md) — Faster-Whisper transcription
5.5. [hallucination_correction](features/hallucination_correction.md) — strip fabricated transcript segments
6. [diarization](features/diarization.md) — speaker attribution
7. [sentence_reconstruction](features/sentence_reconstruction.md) — punctuation + sentence assembly
8. [domain_phonetic_correction](features/domain_phonetic_correction.md) — glossary-grounded term correction
9. [contextual_retrieval](features/contextual_retrieval.md) — chunking + SLM situational context
10. [visual_cue_detection](features/visual_cue_detection.md) — flag chunks needing visual evidence *(video only, conditional)*
11. [frame_extraction](features/frame_extraction.md) — extract + filter/dedup frames
12. [visual_analysis](features/visual_analysis.md) — VLM frame description (Qwen2-VL)
13. [visual_hallucination_correction](features/visual_hallucination_correction.md) — audit visual/text match
14. [visual_context_synthesis](features/visual_context_synthesis.md) — synthesize visual timeline paragraphs
15. [context_unification](features/context_unification.md) — merge audio + visual context
15.5 / 16. [clip_extractor](features/clip_extractor.md) + [rag_storage](features/rag_storage.md) — run concurrently: clip extraction (CPU) and RAG embedding/indexing (GPU)

**Documents** (`BatchDocumentPipeline`, PDF / DOCX): scanner → storage → [document_conversion](features/document_conversion.md) (per-page MarkItDown, OCR for scans) → [document_structuring](features/document_structuring.md) + [contextual_retrieval](features/contextual_retrieval.md) (`execute_document`) → context_unification → rag_storage. Chunks are page- or section-anchored; see [core/anchors.md](core/anchors.md).

**Images** (`BatchImagePipeline`, JPG / PNG / TIFF / WebP / BMP): scanner → storage → [image_description](features/image_description.md) (Qwen2-VL still-image description via [visual_analysis](features/visual_analysis.md), checked against RapidOCR via document_conversion) + contextual_retrieval (`execute_image`, no model) → context_unification → rag_storage. Chunks are image-anchored and their excerpt is labelled a machine description.

**Query time** (not part of batch ingestion):

- Standard search → [rag_retrieval](features/rag_retrieval.md) (hybrid dense + sparse search)
- Verified search → rag_retrieval → [retrieval_hallucination_scrubber](features/retrieval_hallucination_scrubber.md)
- Chat → rag_retrieval → retrieval_hallucination_scrubber → [qa_generation](features/qa_generation.md)

See [orchestration.md](orchestration.md) for the full Phase/Pipeline inventory and [engine.md](engine.md) for how `OnyxEngine` exposes all of this.

## Core infrastructure

- [core/database.md](core/database.md) — SQLAlchemy engine, session management
- [core/jobs.md](core/jobs.md) — job lifecycle/audit-trail tracking
- [core/model_manager.md](core/model_manager.md) — VRAM-budget-driven model load/unload orchestrator
- [core/anchors.md](core/anchors.md) — where a chunk sits in its source (time, page, section) and how citations print it
- [core/language.md](core/language.md) — language packs: every language-specific rule, per appliance language (en, sr, sl)

## Composition layer

- [engine.md](engine.md) — `OnyxBootstrapper` (DI root) and `OnyxEngine` (top-level facade)
- [orchestration.md](orchestration.md) — Phase and Pipeline inventory
- [frontend.md](frontend.md) — the PySide6 desktop app (`frontend/`), which drives `OnyxEngine` in-process
- [deploy.md](deploy.md) — hosted demo: gateway VPS + on-demand Vast.ai GPU box streaming the desktop app over noVNC

## Features (`app/features/`)

All 20 features, in the same order as the pipeline table above:

[scanner](features/scanner.md) · [storage](features/storage.md) · [audio_extractor](features/audio_extractor.md) · [vad](features/vad.md) · [domain_adaptation](features/domain_adaptation.md) · [transcription](features/transcription.md) · [hallucination_correction](features/hallucination_correction.md) · [diarization](features/diarization.md) · [sentence_reconstruction](features/sentence_reconstruction.md) · [domain_phonetic_correction](features/domain_phonetic_correction.md) · [contextual_retrieval](features/contextual_retrieval.md) · [visual_cue_detection](features/visual_cue_detection.md) · [frame_extraction](features/frame_extraction.md) · [visual_analysis](features/visual_analysis.md) · [visual_hallucination_correction](features/visual_hallucination_correction.md) · [visual_context_synthesis](features/visual_context_synthesis.md) · [context_unification](features/context_unification.md) · [clip_extractor](features/clip_extractor.md) · [rag_storage](features/rag_storage.md) · [rag_retrieval](features/rag_retrieval.md) · [retrieval_hallucination_scrubber](features/retrieval_hallucination_scrubber.md) · [qa_generation](features/qa_generation.md)

## Legacy (dead code, kept for reference)

- [legacy_features.md](legacy_features.md) — `asr_error_correction`, `hallucination_scrubber`, `intelligence_bridge`, `nlp_router`, `scene_detector`, each with its confirmed successor

## Dev tools

- [tools/glossary_miner.md](tools/glossary_miner.md) — offline Wikidata glossary mining, feeds `domain_adaptation`/`domain_phonetic_correction`
