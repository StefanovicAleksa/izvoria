# orchestration

`app/orchestration/`

The composition layer between individual features and the top-level `OnyxEngine`: `Phase` classes wrap a single feature's `service` API with job tracking and (for GPU-bound stages) model preload/unload; `Pipeline` classes chain Phases into full workflows.

## Phases (`service/phases/`)

One `Phase` class per pipeline stage — each wraps a feature's `service/*_api.py` call with `JobManager.run_job(...)` and, for GPU-bound stages, a `preload_model()` method that fires an async preload via `AsyncModelService`/`BackgroundModelWorker`, resolved later inside `execute()`, then explicitly unloaded (`trigger_unload(..., flush=True)`) in a `finally` block. This preload-ahead pattern overlaps the next model's load with the current phase's I/O-bound work.

| # | Phase file | Class | What it wraps |
|---|---|---|---|
| 1 | `scan_phase.py` | `ScanPhase` | Directory scan (`scanner`) |
| 2 | `ingest_phase.py` | `IngestPhase` | Registers discovered files as `Source` rows (`storage`) |
| 3 | `extract_audio_phase.py` | `ExtractAudioPhase` | FFmpeg audio extraction (`audio_extractor`) |
| 4 | `vad_phase.py` | `VadPhase` | Voice activity detection (`vad`) |
| 4.5 | `domain_adaptation_phase.py` | `DomainAdaptationPhase` | Domain glossary/hotword generation (`domain_adaptation`) |
| 5 | `transcribe_phase.py` | `TranscribePhase` | Faster-Whisper transcription (`transcription`) |
| 5.5 | `hallucination_correction_phase.py` | `HallucinationCorrectionPhase` | Transcript hallucination filtering (`hallucination_correction`) |
| 6 | `diarize_phase.py` | `DiarizePhase` | Speaker diarization (`diarization`) |
| 7 | `sentence_reconstruction_phase.py` | `SentenceReconstructionPhase` | Punctuation/sentence rebuilding (`sentence_reconstruction`) |
| 8 | `domain_phonetic_correction_phase.py` | `DomainPhoneticCorrectionPhase` | Qwen-based term correction (`domain_phonetic_correction`) |
| 9 | `contextual_retrieval_phase.py` | `ContextualRetrievalPhase` | Chunking + SLM contextualization (`contextual_retrieval`) |
| 10 | `visual_cue_detection_phase.py` | `VisualCueDetectionPhase` | Flags chunks needing visual evidence (`visual_cue_detection`) |
| 11 | `frame_extraction_phase.py` | `FrameExtractionPhase` | FFmpeg frame extraction + filtering (`frame_extraction`) |
| 12 | `visual_analysis_phase.py` | `VisualAnalysisPhase` | VLM frame description (`visual_analysis`) |
| 13 | `visual_hallucination_correction_phase.py` | `VisualHallucinationCorrectionPhase` | SLM audits visual/text match (`visual_hallucination_correction`) |
| 14 | `visual_context_synthesis_phase.py` | `VisualContextSynthesisPhase` | Synthesizes visual timeline paragraphs (`visual_context_synthesis`) |
| 15 | `context_unification_phase.py` | `ContextUnificationPhase` | Merges text + visual context (`context_unification`) |
| 15.5 | `clip_extraction_phase.py` | `ClipExtractionPhase` | Extracts physical `.mp4` clips (`clip_extractor`) |
| 16 | `rag_indexing_phase.py` | `RagIndexingPhase` | Embeds + writes vectors (`rag_storage`) |
| — | `rag_retrieval_phase.py` | `RagRetrievalPhase` | Live hybrid search execution (`rag_retrieval`) |
| — | `retrieval_scrubber_phase.py` | `RetrievalScrubberPhase` | Live SLM hallucination scrubbing (`retrieval_hallucination_scrubber`) |
| — | `qa_generation_phase.py` | `QAGenerationPhase` | Final LLM answer generation with citations (`qa_generation`) |
| *(dead)* | `hallucination_scrubber_phase.py` | `HallucinationScrubberPhase` | Legacy DeBERTa scrubber — not wired into `bootstrap.py` |
| *(dead)* | `intelligence_bridge_phase.py` | `IntelligenceBridgePhase` | Legacy Phi-3.5 NLP bridge — not wired into `bootstrap.py` |

## Pipelines (`service/pipelines/`)

- **`BatchMediaPipeline`** (`batch_media_pipeline.py`) — the canonical video ingestion workflow. Runs phases 1→9 strictly sequentially, then **conditionally** runs the visual branch (phases 10–14) only if `visual_cue_detection` actually flagged chunks needing visual evidence, then phase 15 (unification), then runs clip extraction (15.5) and RAG indexing (16) **concurrently** via a `ThreadPoolExecutor(max_workers=2)` — safe because clip extraction is pure CPU/ffmpeg work while RAG indexing is the sole GPU consumer at that point.
- **`BatchAudioPipeline`** (`batch_audio_pipeline.py`) — the same spine for audio-only sources: filters by `AUDIO_EXTENSIONS`, skips the entire visual branch (phases 10–14) and clip extraction (no video to clip from).
- **`LiveRetrievalPipeline`** (`live_retrieval_pipeline.py`) — query-time fast path: embed the query, run hybrid search, return results. No SLM verification.
- **`LiveVerifiedRetrievalPipeline`** (`live_verified_retrieval_pipeline.py`) — `LiveRetrievalPipeline` plus a pass through the SLM hallucination scrubber for higher-precision "verified search."
- **`LiveQAPipeline`** (`live_qa_pipeline.py`) — the full "chat with your data" chain: retrieval → SLM scrubber → LLM QA generation. Preloads all three models concurrently ahead of executing each stage, then hands off VRAM by unloading before the next stage's model loads.

All pipelines raise `WorkflowStateError` for state problems and wrap unexpected exceptions as `WorkflowExecutionError`; they short-circuit (returning whatever `pipeline_results` has accumulated so far) as soon as a stage yields zero successful items, rather than crashing — so a mostly-empty result dict is a valid, non-exceptional outcome callers must check for, not necessarily a bug.

## Where it fits

Sits directly below `OnyxEngine` (see `engine.md`) — `OnyxEngine.ingest_videos()`/`ingest_audios()`/`search_standard()`/`search_verified()`/`chat()` each delegate to exactly one of the five Pipelines above.
