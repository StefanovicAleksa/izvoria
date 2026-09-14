# Changelog

This log tracks **feature-level milestones**, not every commit. It's written for humans (and future Claude Code sessions) trying to understand how the system got to its current shape — one entry per merged feature milestone, newest on top. See `CLAUDE.md` for the architecture these entries plug into.

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
