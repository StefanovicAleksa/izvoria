# visual_analysis

`app/features/visual_analysis/`

## What it does

Runs a vision-language model over each extracted frame to produce a text description grounded in the surrounding transcript context — turning raw pixels into text the RAG pipeline can actually index and cite.

## How it does it

`QwenVLMAdapter` loads Qwen2-VL-7B via **native Hugging Face Transformers** (not GGUF/llama.cpp — deliberately, due to a documented, unresolved upstream `llama.cpp` SIGSEGV in the Qwen-VL vision graph allocator; see Gotchas). The checkpoint is 4-bit pre-quantized on disk (bitsandbytes/nf4), and inference is bounded to `_MIN_PIXELS`/`_MAX_PIXELS` so a single frame can't blow the reserved VRAM budget.

`VisualAnalysisAPI.process_chunk` sorts a chunk's vaulted frames chronologically and describes them in multimodal calls of up to `VISION_MAX_IMAGES_PER_CALL` (4) frames. `QwenVLMAdapter.analyze_frames` sends a group of frames plus the chunk's text in one chat turn. When there are several frames, each image is downscaled to `VISION_MULTI_IMAGE_MAX_PIXELS` (768·28·28) so they fit one context window. Decoding is greedy (`do_sample=False`), with up to `VISION_MAX_NEW_TOKENS_PER_FRAME` (160) new tokens per frame.

The prompt keeps the transcript so the model knows what to look for, but its rules treat the spoken words as never being evidence:
- describe only what is visibly present;
- transcribe visible text, labels and numbers exactly;
- never state a value that isn't legible, even if the speaker says it;
- say plainly when a frame doesn't show what the speaker describes.

The model answers with one `Frame N: …` line per frame. `utils/frame_description_parser.parse_frame_descriptions` splits the answer and tolerates preambles, markdown emphasis and bullets. If the model ignores the format, the whole answer is attributed to the first frame and a warning is logged. Frames with an empty description are not stored.

Each `FrameAnalysis` carries its frame's `timestamp_sec`, which visual context synthesis uses to build the timeline. The API is idempotent: chunks that were already analyzed are skipped. It exposes `.manifest`/`.adapter` properties so the orchestration `Phase` can preload the VLM once and reuse it across every chunk.

## Inputs / Outputs

Input: a `ContextualizedChunk` (with associated `ExtractedFrame`s from `frame_extraction`). Output: `ChunkVisualAnalysis` (`analyses: List[FrameAnalysis]`, each with `frame_id`, `description`, `timestamp_sec`).

## Where it fits

Phase 12 (`VisualAnalysisPhase`) — runs after frame extraction, before visual hallucination correction (which checks whether these descriptions actually match the transcript).

## Gotchas / design notes

- **Must stay on native HF Transformers, not llama.cpp/GGUF.** `llama.cpp`'s `mtmd` vision encoder has a confirmed, unresolved upstream SIGSEGV specific to the Qwen-VL model family — see `integration_tests/features/visual_analysis/test_vlm_multimodal_diagnostics.py` for the documented regression evidence. Don't "simplify" this adapter onto llama.cpp for consistency with the other LLM adapters in the codebase.
- Pre-quantizing to disk (rather than quantizing at load time) keeps the on-disk size representative of real VRAM usage, which `ModelManagerOrchestrator`'s admission control relies on.
