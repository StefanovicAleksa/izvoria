# contextual_retrieval

`app/features/contextual_retrieval/`

## What it does

Implements Anthropic's "Contextual Retrieval" pattern — chunks the corrected transcript into ~250-word RAG-sized chunks and uses an SLM to prepend a short situational-context string to each chunk before embedding, improving retrieval precision. This feature also owns the central `ContextualizedTranscript`/`ContextualizedChunk` entities that the entire visual branch (`visual_cue_detection`, `visual_hallucination_correction`, `visual_context_synthesis`) reads and mutates.

## How it does it

`ContextualizerAPI.execute()` takes a `DomainPhoneticCorrectedTranscript`, is idempotent, and calls `build_rag_chunks()` (`utils/chunker.py`) — a deterministic sliding-window grouper over `ReconstructedSentence` objects (the atomic unit, preserving timestamps) targeting `max_words=250` with 1-sentence overlap between chunks, always advancing by at least one sentence even if a single sentence exceeds `max_words` (guards against an infinite loop).

Both the chunk text and the document the SLM reads are rendered as speaker turns by `utils/transcript_formatter.format_speaker_turns` — `[Speaker_0 @ 00:12:31]: First sentence. Second sentence.` — merging consecutive same-speaker sentences, so "who said what, when" reaches retrieval and answers. The labelled chunk text is stored as `ContextualizedChunk.original_text`. The SLM reads the same turns without brackets or timestamps (`Speaker_0: ...`, via `include_timestamps=False`) — with the timestamped form, Llama 3.2 3B tended to answer by echoing a label — and `SLMContextAdapter.generate_context` strips any leading speaker label the model still echoes.

Document sizing: if the whole labelled transcript fits `CONTEXTUAL_RETRIEVAL_N_CTX - CONTEXTUAL_RETRIEVAL_RESERVED_TOKENS` (measured with `ISLMContextAdapter.count_tokens`), every chunk uses it as its document — one stable prefix the llama.cpp KV cache reuses. Otherwise `utils/document_window.build_document_sections` splits the transcript into overlapping sections (each filled to the budget, stepping by half a budget) and `assign_section` gives each chunk the section it sits in most centrally, so the prompt prefix only changes when the section does. A long transcript used to overflow the context window and fail the whole source.

For each raw chunk it calls `ISLMContextAdapter.generate_context(model, full_document_text, chunk_text)`. The concrete `SLMContextAdapter` runs Llama 3.2 1B/3B or Qwen 2.5 via `llama_cpp` with a large `n_ctx` (`CONTEXTUAL_RETRIEVAL_N_CTX`, default 32768) and `flash_attn=True` "to prevent OOM on 32k contexts", `temperature=0.0`, `max_tokens=150`, plus defensive stripping of "Here is the context:" preambles the model sometimes adds. The model is lazy-imported inside a static method specifically to avoid a C-extension collision with PyTorch during application startup.

## Inputs / Outputs

Input: `DomainPhoneticCorrectedTranscript`. Output: `ContextualizedTranscript` (`chunks: List[ContextualizedChunk]`). `ContextualizedChunk.combined_text` (`context_string + "\n\n" + original_text`) is what actually gets embedded and full-text-indexed downstream.

## Where it fits

Phase 9 (`ContextualRetrievalPhase`) — the chunking boundary between the linear audio-correction pipeline and everything after it (the visual branch, unification, RAG indexing all operate on `ContextualizedChunk`s).

## Gotchas / design notes

- Entities are frozen (immutable) Pydantic models. `IContextualRepository.update()` exists specifically so later features (notably `visual_cue_detection`) can JSONB-overwrite the `chunks` field in place (e.g. to set `needs_visual`) without replacing the whole transcript row.
