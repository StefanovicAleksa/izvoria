# rag_retrieval

`app/features/rag_retrieval/`

## What it does

Answers a query against the indexed knowledge base using hybrid (dense + sparse) search — this is the stable retrieval contract that anything built on top of RAG (verified search, QA generation, and the upcoming agent) calls into.

## How it does it

`RetrievalAPI.search(query: SearchQuery)` builds the same `RAG_EMBEDDING` model manifest used at indexing time, acquires the embedder, embeds the query text, then **immediately releases/flushes** it before running the DB query — a strict load→use→unload pattern, since the next pipeline stage (scrubber or LLM) needs that VRAM back. It then calls `PostgresSearchRepository.execute_hybrid_search()`.

`execute_hybrid_search` runs a single SQL query combining a dense pgvector `<=>` cosine-distance CTE and a sparse `ts_rank_cd`/`plainto_tsquery` CTE, `FULL OUTER JOIN`s the two result sets, and merges them with **Reciprocal Rank Fusion** (`1/(rrf_k+rank)` summed per side, `rrf_k` from `settings.RAG_RRF_K_CONSTANT`), also left-joining `video_clips` to resolve a physical clip id for each result where one exists.

## Inputs / Outputs

The stable contract for anything downstream: `IRetrievalRepository.execute_hybrid_search(query_vector, query_text, limit, source_id_filter) -> List[RetrievedChunk]`, and the higher-level `RetrievalAPI.search(SearchQuery) -> SearchResult`. `RetrievedChunk` = `{chunk_id, source_id, text_payload, start_sec, end_sec, relevance_score, extracted_file_id}`.

## Where it fits

Used at query time by `LiveRetrievalPipeline` (standard search), and as the first stage of `LiveVerifiedRetrievalPipeline` and `LiveQAPipeline` (verified search / chat). Exposed at the top level as `OnyxEngine.search_standard()`.

## Gotchas / design notes

This is the natural, already-stable hook for building an agent on top of RAG — no need to reimplement hybrid search, just call `RetrievalAPI.search()` (or `IRetrievalRepository.execute_hybrid_search()` one layer down) and, if hallucination-filtered context is needed, follow it with `retrieval_hallucination_scrubber`.
