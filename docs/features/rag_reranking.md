# rag_reranking

`app/features/rag_reranking/`

## What it does

Second-stage relevance ranking for live search — the reranking step of Anthropic's Contextual Retrieval. Hybrid retrieval returns a broad RRF-fused candidate pool; a cross-encoder then reads the query and each candidate **together** and keeps the best `query.limit` chunks. Scoring query and chunk jointly is far more precise than fusing independent dense and sparse ranks, which is why the paper reports reranking as the largest single gain on top of contextual embeddings + contextual BM25.

## How it does it

- **`domain/interfaces.py`**: `IRerankerAdapter(IModelLoader).score(model, query, texts) -> List[float]` — one score per text, same order, higher is more relevant.
- **`data/cross_encoder_adapter.py` — `CrossEncoderRerankerAdapter`**: wraps `sentence_transformers.CrossEncoder` (bge-reranker-v2-m3), fp16 on CUDA, `local_files_only=True`, `max_length=1024`, batch size 16. The model import runs inside the shared `block_torchcodec_import()` guard.
- **`service/rerank_api.py` — `RerankAPI.rerank(query, chunks, top_n, release_after=True)`**: acquires the `RAG_RERANKING` model through the orchestrator (`wait_for_clear=True`), scores each chunk's `text_payload` (the contextualized text), sorts by score (stable, so ties keep RRF order), sets `RetrievedChunk.rerank_score`, and returns the top `top_n`. Always releases in `finally` when `release_after=True`.
- **`exceptions/`** (`RerankingError`, `RerankerModelLoadError`, `RerankInferenceError`) + **`utils/error_mapper.py`**.

## Inputs / Outputs

Input: the query string + the candidate `List[RetrievedChunk]` from `rag_retrieval`. Output: at most `top_n` `RetrievedChunk`s, best first, each with `rerank_score` set.

## Where it fits

`RagRerankingPhase` (`app/orchestration/service/phases/rag_reranking_phase.py`) runs directly after `RagRetrievalPhase` in all three live pipelines: standard search (retrieve → rerank), verified search (retrieve → rerank → scrub) and chat (retrieve → rerank → scrub → generate).

## Configuration

- `RAG_RERANKER_MODEL_PATH` (default `models/bge-reranker-v2-m3`) — download `BAAI/bge-reranker-v2-m3` there.
- `RAG_RERANKER_VRAM_REQUIREMENT_MB` (default 1500).
- `RAG_RETRIEVAL_CANDIDATES` (default 40) — size of the candidate pool retrieval hands to the reranker.

## Gotchas / design notes

- **Optional by design.** `RerankAPI.is_available()` checks the weights exist; without them `rerank()` logs a warning and returns `chunks[:top_n]` in RRF order, and the phase skips its preload. Search keeps working on an appliance that hasn't installed the model.
- Scores are only used for ordering; they are not calibrated probabilities and there is no score threshold — filtering irrelevant chunks remains the scrubber's job.
