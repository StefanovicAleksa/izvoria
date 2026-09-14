# Changelog

This log tracks **feature-level milestones**, not every commit. It's written for humans (and future Claude Code sessions) trying to understand how the system got to its current shape — one entry per merged feature milestone, newest on top. See `CLAUDE.md` for the architecture these entries plug into.

## Review fixes: stability, retrieval grounding, audio correctness, throughput

Fixes from the full review of RAG retrieval/storage and the audio/video ingestion pipelines, delivered in four batches on `fix/review-pipeline-retrieval`. Each bullet is one change: what changed, why, and where.

### Batch 1 — Stability & data integrity

- **Model residency is keyed on `(model_type, model_path)`** (`core/model_manager/service/orchestrator.py`). The same GGUF is loaded by several adapters with different settings (e.g. Llama 3.2 3B at `n_ctx` 4096 for domain classification and 32768 for contextual retrieval), so a resident instance could previously be handed to the wrong feature. The idempotency check also now runs under the orchestrator lock instead of iterating `_active_models` lock-free.
- **`acquire_model(wait_for_clear=True)` has a timeout** (`MODEL_WAIT_FOR_CLEAR_TIMEOUT_SEC`, default 120s) and raises the new `ModelAcquireTimeoutError` (`core/model_manager/exceptions/timeout.py`). A leaked model used to park the caller — and the single preload worker thread — forever.
- **Unresolved preloads are discarded, not leaked.** `BackgroundModelWorker.discard_preload` / `AsyncModelService.discard_preload` cancel a queued preload or release the model as soon as a running one finishes. Every GPU phase clears its future once resolved and exposes `discard_preload()`, and all five pipelines call it for every phase in a `finally`. Previously a failed live query (e.g. an empty chat prompt) left the scrubber/QA preloads loading with nobody to release them, hanging every later search and chat until restart; the batch pipelines had the same leak on every early return.
- **The engine evicts all models after ingestion and after any failed live call** (`OnyxEngine._cleanup_after_ingestion`, `_release_models_after_failure`, via `AsyncModelService.release_all`). Before, cleanup only ran `gc` and orchestrator-tracked models stayed resident.
- **Blank search prompts are rejected at validation** (`SearchQuery.prompt` is stripped and requires `min_length=1`), instead of failing inside the embedder after a model was already loaded.
- **Clip extraction cut the wrong length** (`clip_extractor/data/ffmpeg_video_adapter.py`). With input seeking (`-ss` before `-i`), FFmpeg treats an output `-to` as a *duration*, so a chunk at 3600–3700s produced a ~3700s clip (verified on `integration_tests/resources/test_video.mp4`: a 3→5s request produced 5s). The command now passes `-t (end - start)`.
- **Clips vaulted before a mid-run failure are deleted** (`ClipExtractorAPI` + new `StorageAPI.delete_raw_file` / `IStorageRepository.delete_file`). They had no DB rows, so they became orphaned files and were re-extracted on the next run.
- **Job output is actually persisted** (`core/jobs/data/repositories.py`). `finish_job` wrote to the unmapped attribute `output_result`, so nothing was ever stored; it now writes the `output_params` column, which the `Job` entity reads back. `JobManager` stores a compact JSON-safe summary (scalar dict entries without `_`-prefixed transient keys, or `{result_type, result_id}`) instead of `str()` of whole transcripts.
- **RAG indexing is atomic and failures are real failures** (`rag_storage`). Chunks are embedded first, then `replace_rag_chunks` deletes and inserts in one transaction — previously the old index was deleted before embedding, so a failure left the source with no chunks. `index_transcript` raises typed `RAGStorageError`s instead of returning `success=False` (which marked the job COMPLETED), and `map_storage_error` always returns a feature exception with the original attached.
- **Silent-track fallback only for media with no audio stream** (`audio_extractor/data/ffmpeg_adapter.py`). Streams are counted with `ffprobe`; previously any FFmpeg stderr containing "Invalid argument" was treated as "no audio", silently indexing a video with an empty transcript.
- **Diarization is crash-safe and fallback use is recorded** (`diarization`). The map (the idempotency marker) is saved after the aligned transcript is stored, so a crash in between no longer skips alignment forever. Clustering fallbacks log a warning instead of `print` and set `DiarizationMap.fallback_applied` (new column).

## Unreleased / In progress

- **Environment/hardware diagnostics**: new integration tests documenting two subprocess-isolated regressions — cross-feature `sys.modules["torchcodec"]` leakage in long-lived processes, and the confirmed llama.cpp SIGSEGV in the Qwen-VL vision encoder (native HF Transformers remains the required path for vision). See "Known gotchas" in `CLAUDE.md`.
- **Config/dependency updates** in progress alongside the diagnostics above.
- **Next milestone: the RAG-grounded conversational agent** — moving beyond the existing single-pass retrieve → verify → generate flow (`LiveQAPipeline`) toward multi-turn, tool-using interaction built on `RetrievalAPI`, `RetrievalScrubberAPI`, and `QAGenerationAPI`.

## QA Generation integrated into orchestration & engine

`OnyxEngine.chat()` now runs the full retrieve → verify → generate chain via `LiveQAPipeline`: hybrid RAG retrieval, hallucination-scrubbed filtering, then Gemma-generated answers with citations parsed back to source chunks.

## QA Generation feature

New `qa_generation` feature: `GemmaGenerationAdapter` loads a local Gemma GGUF model via `llama-cpp-python` (`n_gpu_layers=-1`, 8K context, flash-attn). `QAGenerationAPI.generate(query, retrieved_chunks)` builds a strict context block from retrieved chunks, runs chat completion, and parses citations back out of the generated text to their source chunk/document ids — this is the boundary that enforces "never answer outside the retrieved context."

## Verified search integrated into the engine

`retrieval_hallucination_scrubber` wired into `LiveVerifiedRetrievalPipeline` and exposed as `OnyxEngine.search_verified()`. "Verified search" = standard hybrid retrieval followed by an SLM relevance judge (`IRetrievalScrubberAdapter.evaluate_relevance`) that drops chunks it judges irrelevant or likely to produce a hallucinated answer, before anything reaches generation.

## RAG retrieval + contextual retrieval pipelines

`rag_retrieval` and `contextual_retrieval` features and their pipelines created and integrated. `RetrievalAPI.search(SearchQuery)` performs hybrid search: dense (pgvector cosine distance) + sparse (Postgres full-text `ts_rank_cd`) combined via Reciprocal Rank Fusion (`RAG_RRF_K_CONSTANT`), returning `RetrievedChunk` objects with text, timing, and a link back to the source clip.

## Retrieval hallucination scrubber

New `retrieval_hallucination_scrubber` feature: unit- and integration-tested SLM-based relevance judge for filtering irrelevant or likely-hallucinated retrieved chunks before they reach an LLM — the first half of what later becomes "verified search."

## Core ingestion pipeline (undated — pre-existing at time of writing)

The batch media/audio ingestion pipeline (`BatchMediaPipeline` / `BatchAudioPipeline`), built stage by stage as features under `app/features/` and wired together by `app/orchestration/`:

- Media ingestion & extraction (FFmpeg, SHA-256 dedup)
- Voice Activity Detection (NVIDIA NeMo MarbleNet)
- Domain adaptation / "Scout" (Faster-Whisper Tiny draft + DeBERTa v3 classifier + hotword extraction — detects Legal/Medical/General)
- High-fidelity transcription (Faster-Whisper Large v3, hotword-biased)
- Speaker diarization (NVIDIA NeMo TitaNet)
- Hallucination correction, sentence reconstruction (deterministic punctuation classification + speaker-aware assembly), domain phonetic correction (glossary-grounded local LLM correction with an audit trail)
- Visual branch: visual cue detection → frame extraction → visual analysis (Qwen2-VL) → visual hallucination correction → visual context synthesis
- Context unification and RAG indexing (embedding via BGE-M3 + storage into the hybrid pgvector/full-text table), run concurrently with clip extraction

Underpinning all of it: `app/core/model_manager`'s `ModelManagerOrchestrator` — a VRAM-budget-driven admission-control and swap-on-acquire model registry, built specifically to fit this whole pipeline through a single 12GB GPU on dev hardware by loading and unloading each model just-in-time.
