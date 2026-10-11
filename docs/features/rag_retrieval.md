# rag_retrieval

`app/features/rag_retrieval/`

## What it does

Answers a query against the indexed knowledge base using hybrid (dense + sparse) search — this is the stable retrieval contract that anything built on top of RAG (verified search, QA generation, and the upcoming agent) calls into.

## How it does it

`RetrievalAPI.search(query: SearchQuery)` builds the same `RAG_EMBEDDING` model manifest used at indexing time, acquires the embedder, embeds the query text, then **immediately releases/flushes** it before running the DB query — a strict load→use→unload pattern, since the next pipeline stage (scrubber or LLM) needs that VRAM back. It then calls `PostgresSearchRepository.execute_hybrid_search()` for a candidate pool of `max(query.limit, RAG_RETRIEVAL_CANDIDATES)` chunks (default 40); narrowing that pool to `query.limit` is `rag_reranking`'s job.

`execute_hybrid_search` runs a single SQL query combining a dense pgvector `<=>` cosine-distance CTE and a sparse full-text CTE, `FULL OUTER JOIN`s the two result sets, and merges them with **Reciprocal Rank Fusion** (`1/(rrf_k+rank)` summed per side, `rrf_k` from `settings.RAG_RRF_K_CONSTANT`), also left-joining `video_clips` to resolve a physical clip id for each result where one exists.

The sparse leg uses **OR semantics**: `plainto_tsquery` output is rewritten from `' & '` to `' | '` (keeping Postgres stemming and stop-word removal), so any matching term counts, and results are ranked with length-normalized `ts_rank(fts_document, tsq, 1)`. `plainto_tsquery` alone ANDs every lexeme, which made natural-language questions ("what did the doctor say about the MRI results") match almost nothing. A question made only of stop words produces no sparse query. The text search configuration comes from `settings.FTS_LANGUAGE_CONFIG`.

## Inputs / Outputs

The stable contract for anything downstream: `IRetrievalRepository.execute_hybrid_search(query_vector, query_text, limit, source_id_filter) -> List[RetrievedChunk]`, and the higher-level `RetrievalAPI.search(SearchQuery) -> SearchResult`. `RetrievedChunk` = `{chunk_id, source_id, text_payload, raw_text, speaker_labels, start_sec, end_sec, relevance_score, rerank_score, extracted_file_id}` — `text_payload` is the contextualized text that was indexed, `raw_text` the verbatim speaker-labelled transcript an answer model may read, `rerank_score` is set by `rag_reranking`. `SearchQuery` rejects blank prompts and defaults `limit` to 10 (max 20).

## Where it fits

The first stage of all three live pipelines — `LiveRetrievalPipeline` (standard search), `LiveVerifiedRetrievalPipeline` and `LiveQAPipeline` (verified search / chat) — always followed by `rag_reranking`. Exposed at the top level as `IzvoriaEngine.search_standard()`.

## Gotchas / design notes

This is the natural, already-stable hook for building an agent on top of RAG — no need to reimplement hybrid search, just call `RetrievalAPI.search()` (or `IRetrievalRepository.execute_hybrid_search()` one layer down) and, if hallucination-filtered context is needed, follow it with `retrieval_hallucination_scrubber`.

## Language

One index holds every language, and each chunk carries its own text search configuration (`vector_chunks.fts_config`): `english`, `serbian` (Snowball; stems Cyrillic, Latin, ekavian and ijekavian to one stem) or `simple` (Slovenian; Postgres ships no stemmer).

- **The sparse leg builds one tsquery per configuration** in `settings.SEARCH_FTS_CONFIGS` (`utils/sparse_query.py`, bound as the aligned arrays `:fts_configs` / `:query_texts`). Each chunk is matched only against the query built with its own configuration (`JOIN sparse_query sq ON v.fts_config = sq.cfg`), so the GIN index stays usable. RRF fuses ranks, so the per-configuration `ts_rank` scales never need to be comparable.
- **`simple` keeps stop words**, so `simple` gets the question minus every pack's stop words. Otherwise the "the", "a" or "je" of a question in any language matched nearly every chunk it indexed. No served pack uses `simple` since Slovenian became legacy; the handling stays for a language Postgres has no stemmer for. Stemmed configurations drop their own stop words.
- **Cross-language retrieval comes from the dense leg**: BGE-M3 is multilingual, so an English question finds a Serbian chunk. The reranker (bge-reranker-v2-m3) is multilingual too.
- **The dense leg's best always reach the reranker** (`dense_keep`, `RAG_DENSE_CANDIDATES_KEPT` = 10, passed by `RetrievalAPI.search` / `search_batch` for their reranker pools). Asked in another language than the material, the keyword leg can only match chunks in the question's language, and under RRF every chunk with a keyword match outscores the dense leg's #1 without one. Measured on the cross-language eval, a Serbian question put the right English chunk at fused rank ~37, although BGE-M3 ranked it top 3. The sweep asks for exactly `per_source_limit` chunks and passes no `dense_keep`, so a report's evidence bank doesn't grow.
- A query is embedded and keyword-matched in the index's scripts: `_search_text` applies every pack's script rule, so a Serbian query typed in Cyrillic searches the Latin Serbian chunks whichever language is active. Results stay keyed by the prompt as written.

