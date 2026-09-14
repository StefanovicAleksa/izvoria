# contextual_retrieval

`app/features/contextual_retrieval/`

## What it does

Implements Anthropic's "Contextual Retrieval" pattern — chunks the corrected transcript into ~250-word RAG-sized chunks and uses an SLM to prepend a short situational-context string to each chunk before embedding, improving retrieval precision. This feature also owns the central `ContextualizedTranscript`/`ContextualizedChunk` entities that the entire visual branch (`visual_cue_detection`, `visual_hallucination_correction`, `visual_context_synthesis`) reads and mutates.

## How it does it

`ContextualizerAPI.execute()` takes a `DomainPhoneticCorrectedTranscript`, is idempotent, builds the full document text by joining all sentences, and calls `build_rag_chunks()` (`utils/chunker.py`) — a deterministic sliding-window grouper over `ReconstructedSentence` objects (the atomic unit, preserving timestamps) targeting `max_words=250` with 1-sentence overlap between chunks, always advancing by at least one sentence even if a single sentence exceeds `max_words` (guards against an infinite loop).

For each raw chunk it calls `ISLMContextAdapter.generate_context(model, full_document_text, chunk_text)`. The concrete `SLMContextAdapter` runs Llama 3.2 1B/3B or Qwen 2.5 via `llama_cpp` with a notably large `n_ctx=32768` (to hold the whole document at once) and `flash_attn=True` "to prevent OOM on 32k contexts", `temperature=0.0`, `max_tokens=150`, plus defensive stripping of "Here is the context:" preambles the model sometimes adds. The model is lazy-imported inside a static method specifically to avoid a C-extension collision with PyTorch during application startup.

## Inputs / Outputs

Input: `DomainPhoneticCorrectedTranscript`. Output: `ContextualizedTranscript` (`chunks: List[ContextualizedChunk]`). `ContextualizedChunk.combined_text` (`context_string + "\n\n" + original_text`) is what actually gets embedded and full-text-indexed downstream.

## Where it fits

Phase 9 (`ContextualRetrievalPhase`) — the chunking boundary between the linear audio-correction pipeline and everything after it (the visual branch, unification, RAG indexing all operate on `ContextualizedChunk`s).

## Gotchas / design notes

- Entities are frozen (immutable) Pydantic models. `IContextualRepository.update()` exists specifically so later features (notably `visual_cue_detection`) can JSONB-overwrite the `chunks` field in place (e.g. to set `needs_visual`) without replacing the whole transcript row.
