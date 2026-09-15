# visual_analysis

`app/features/visual_analysis/`

## What it does

Runs a vision-language model over each extracted frame to produce a text description grounded in the surrounding transcript context — turning raw pixels into text the RAG pipeline can actually index and cite.

## How it does it

`QwenVLMAdapter` loads Qwen2-VL-7B via **native Hugging Face Transformers** (not GGUF/llama.cpp — deliberately, due to a documented, unresolved upstream `llama.cpp` SIGSEGV in the Qwen-VL vision graph allocator; see Gotchas). The checkpoint is 4-bit pre-quantized on disk (bitsandbytes/nf4), and inference is bounded to `_MIN_PIXELS`/`_MAX_PIXELS` so a single frame can't blow the reserved VRAM budget.

`VisualAnalysisAPI.process_chunk` sorts a chunk's vaulted frames chronologically and reads them in multimodal calls of up to `VISION_MAX_IMAGES_PER_CALL` (4) frames.

**One call.** `QwenVLMAdapter.analyze_frames` sends a group of frames plus the chunk's text in one chat turn.
- When there are several frames, each image is downscaled to `VISION_MULTI_IMAGE_MAX_PIXELS` (768·28·28) so they fit one context window.
- Decoding is greedy (`do_sample=False`), with up to `VISION_MAX_NEW_TOKENS_PER_FRAME` (200) new tokens per frame.

**Fields, not prose.** The prompt keeps the transcript so the model knows what to look for, but the model reports fields instead of writing a description:
- `SHOWS`: what is on screen;
- `TEXT`: quoted legible text, title first, at most 8 items;
- `MARKS`: drawn arrows, lines, boxes and callouts, and what they point at;
- `LINK`: which spoken reference the text or marks show.

`utils/frame_description_parser.parse_frame_observations` splits the answer per `Frame N:` marker and reads the fields into a `FrameObservation`. A section without a `SHOWS` field gives `None`.

**Grounding** (`utils/grounding_check.py`, deterministic). A frame keeps a description only if it carries visible evidence.
- `ground_observation` drops a frame that shows only the speaker. It also removes:
  - website addresses, watermarks, video player chrome ("16 / 31", "PREV") and outline sidebars (3+ short numbered entries);
  - repeated items;
  - text items holding a number whose content words sit together in one stretch of the speech ("160.84 if we can get back above 63"), and bare numbers the speaker says ("142.04" listed for a chart whose axis is unreadable);
  - generic marks ("arrows pointing to different parts of the chart");
  - marks naming a spoken number that the frame's own text doesn't show.
- A `LINK` with a number the speaker never said is removed.
- `drop_marks_copied_from_other_frames` clears marks that point at text another frame in the same call reported.
- `drop_items_already_reported` removes text an earlier frame of the chunk already reported, which removes text a multimodal call copied from the first frame.
- A frame with no legible text and no specific mark left gets no description at all.

**Rendering** (`utils/observation_renderer.render_observation`). The stored description is written from the fields, with screen and speech kept in separate sentences:

`Slide with on-screen text "BoE REVERSAL 22/3/18"; "Result 7-2 for hike BUT". Marked on screen: arrow at the spike. Speaker's reference: the 7-2 vote.`

Each `FrameAnalysis` carries its frame's `timestamp_sec`, which visual context synthesis uses to build the timeline. The API is idempotent: chunks that were already analyzed are skipped. It exposes `.manifest`/`.adapter` properties so the orchestration `Phase` can preload the VLM once and reuse it across every chunk.

## Inputs / Outputs

Input: a `ContextualizedChunk` (with associated `ExtractedFrame`s from `frame_extraction`). Output: `ChunkVisualAnalysis` (`analyses: List[FrameAnalysis]`, each with `frame_id`, rendered `description`, `timestamp_sec`; frames without visible evidence are absent).

## Where it fits

Phase 12 (`VisualAnalysisPhase`) — runs after frame extraction, before visual hallucination correction (which checks whether these descriptions actually match the transcript).

## Gotchas / design notes

- **Must stay on native HF Transformers, not llama.cpp/GGUF.** `llama.cpp`'s `mtmd` vision encoder has a confirmed, unresolved upstream SIGSEGV specific to the Qwen-VL model family — see `integration_tests/features/visual_analysis/test_vlm_multimodal_diagnostics.py` for the documented regression evidence. Don't "simplify" this adapter onto llama.cpp for consistency with the other LLM adapters in the codebase.
- Pre-quantizing to disk (rather than quantizing at load time) keeps the on-disk size representative of real VRAM usage, which `ModelManagerOrchestrator`'s admission control relies on.
- **Why fields instead of a description.** Asked to "describe the frame", the 4-bit 7B model paraphrased the speech over slides whose text was perfectly legible. On the 74-minute video, 11 of 20 stored descriptions were paraphrases and 9 were indexed.
  - Asking for fields and rendering them deterministically gave 16 anchored descriptions and 0 paraphrases.
  - Evaluate changes with `scripts/eval/visual_eval.py` against `scripts/eval/visual_labels.json`.
- **Known limits.**
  - In a multi-frame call the model sometimes attributes one frame's text to another frame of the same chunk. The text is on screen during the chunk, but its frame timestamp can be off. One image per call avoided this, but was slower (217 s vs 165 s) and added other errors.
  - Short OCR slips survive ("WVPOC" read as "W1").

