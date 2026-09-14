# visual_analysis

`app/features/visual_analysis/`

## What it does

Runs a vision-language model over each extracted frame to produce a text description grounded in the surrounding transcript context — turning raw pixels into text the RAG pipeline can actually index and cite.

## How it does it

`QwenVLMAdapter` loads Qwen2-VL-7B via **native Hugging Face Transformers** (not GGUF/llama.cpp — deliberately, due to a documented, unresolved upstream `llama.cpp` SIGSEGV in the Qwen-VL vision graph allocator; see Gotchas). The checkpoint is 4-bit pre-quantized on disk (bitsandbytes/nf4), and inference is bounded to `_MIN_PIXELS`/`_MAX_PIXELS` so a single frame can't blow the reserved VRAM budget.

`VisualAnalysisAPI.process_chunk` pulls extracted frames from the storage vault for a given chunk and runs the VLM per frame with a text-grounded prompt built from the chunk's transcript context plus the original text — the model is explicitly steered by what was actually said, not asked to describe the image in a vacuum. It persists `ChunkVisualAnalysis`/`FrameAnalysis` records, and is idempotent (skips chunks already analyzed). It exposes `.manifest`/`.adapter` properties specifically so the orchestration `Phase` can batch-preload the VLM once and reuse it across many chunks/frames rather than reloading per chunk — a significant VRAM/latency optimization given how large the model is.

## Inputs / Outputs

Input: a `ContextualizedChunk` (with associated `ExtractedFrame`s from `frame_extraction`). Output: `ChunkVisualAnalysis` (`frame_analyses: List[FrameAnalysis]`).

## Where it fits

Phase 12 (`VisualAnalysisPhase`) — runs after frame extraction, before visual hallucination correction (which checks whether these descriptions actually match the transcript).

## Gotchas / design notes

- **Must stay on native HF Transformers, not llama.cpp/GGUF.** `llama.cpp`'s `mtmd` vision encoder has a confirmed, unresolved upstream SIGSEGV specific to the Qwen-VL model family — see `integration_tests/features/visual_analysis/test_vlm_multimodal_diagnostics.py` for the documented regression evidence. Don't "simplify" this adapter onto llama.cpp for consistency with the other LLM adapters in the codebase.
- Pre-quantizing to disk (rather than quantizing at load time) keeps the on-disk size representative of real VRAM usage, which `ModelManagerOrchestrator`'s admission control relies on.
