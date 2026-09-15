# qa_generation

`app/features/qa_generation/`

## What it does

The final "chat with your data" stage — takes a user's question plus a set of (ideally already-scrubbed) retrieved chunks and generates a natural-language answer strictly grounded in that context, with citations back to the specific source chunks it drew from.

## How it does it

`GemmaGenerationAdapter` loads a local Gemma model in GGUF format via `llama_cpp.Llama` (`n_gpu_layers=-1`, 8K context, flash-attention enabled). `QAGenerationAPI.generate(user_query, retrieved_chunks)` builds a strict context block from the retrieved chunks via `citation_parser.build_context_block` — each chunk's verbatim `raw_text` under a `[Doc N] (HH:MM:SS–HH:MM:SS)` marker the model is instructed to cite. The generated situating context in `text_payload` is deliberately **not** shown: it helps retrieval but isn't source material, and an answer grounded only in it would still have received a valid citation. The system prompt explains the `[Speaker_N @ HH:MM:SS]` labels and forbids inventing names for speakers. The API runs chat completion against that context, and then parses citations back out of the generated text (`parse_citations`), mapping each `[Doc N]` tag to its chunk and carrying the chunk's `source_id` and `extracted_file_id` (its vaulted clip) so the engine can resolve a playable path; the parser itself never sets a path. This citation-parsing step is the mechanism that enforces "every claim must be traceable to a source" — the model is instructed to cite, and the output is then programmatically checked/mapped rather than trusted blindly.

## Inputs / Outputs

Input: `user_query: str`, `retrieved_chunks: List[RetrievedChunk]`. Output: a generated answer plus parsed citations, assembled into `QAResponse`.

## Where it fits

The final stage of `LiveQAPipeline`, after `rag_retrieval`, `rag_reranking` and `retrieval_hallucination_scrubber`. Exposed at the top level as `OnyxEngine.chat()`, which hydrates each citation's `physical_path` exactly as search hydrates result chunks: the extracted clip if there is one, otherwise the ingested source file.

## Gotchas / design notes

This — plus `rag_retrieval` and `retrieval_hallucination_scrubber` — is the existing single-pass retrieve → verify → generate chain (`LiveQAPipeline`). The RAG agent currently being built is expected to either call this chain directly for simple Q&A, or reimplement a version of it with more control (e.g. multi-turn state, tool use) rather than assuming this feature alone is sufficient for agentic behavior.
