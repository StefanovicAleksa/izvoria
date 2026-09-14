# visual_cue_detection

`app/features/visual_cue_detection/`

## What it does

Decides, per transcript chunk, whether the text actually needs visual evidence (e.g. "as shown in the diagram") — a cheap gate so downstream frame extraction and VLM analysis only run on chunks that genuinely reference something visual, instead of processing every chunk of every video.

## How it does it

`VisualCueDetectionAPI.process_transcript()` acquires a model via `ModelManagerOrchestrator` using a `ModelManifest(model_type=ModelType.VISUAL_CUE_DETECTION, ...)`, then loops `transcript.chunks`, calling `IVisualCueDetectorAdapter.detect_cues(model, chunk)` for each. The concrete `QwenDetectorAdapter` runs Qwen 2.5 GGUF via `llama_cpp.Llama` (`n_gpu_layers=-1`, `flash_attn=True`, `n_ctx=2048`), forces a JSON-schema response (`CUE_DETECTION_SCHEMA`) at `temperature=0.0` (reproducible verdicts), and returns `chunk.model_copy(update={"needs_visual": bool})` (chunks are frozen Pydantic models, so mutation always goes through `model_copy`). Results are persisted via `contextual_retrieval`'s `IContextualRepository.update()` (a JSONB overwrite) rather than a dedicated repository of its own. The API marks every judged chunk `cue_evaluated=True` and skips chunks already evaluated. If every chunk is already evaluated, it returns without loading the model or writing anything, so a rerun after a crash doesn't re-judge finished transcripts.

## Inputs / Outputs

Input: `ContextualizedTranscript`. Output: the same transcript with `needs_visual` and `cue_evaluated` set per chunk.

## Where it fits

Phase 10 (`VisualCueDetectionPhase`) — the entry point to the conditional visual branch. `BatchMediaPipeline` only proceeds into frame extraction/visual analysis/etc. if this phase actually flags at least one chunk.

## Gotchas / design notes

- Per-chunk exceptions are wrapped via `map_detection_error` and **re-raised**, not swallowed — explicitly so `JobManager` marks the whole job `FAILED` rather than silently under-detecting visual cues.
- `unload()` explicitly calls `.close()` on the llama.cpp instance before `del` and `torch.cuda.empty_cache()`, a defensive pattern against VRAM leaks specific to llama.cpp bindings.
