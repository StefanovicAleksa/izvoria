# rag_storage

`app/features/rag_storage/`

## What it does

Turns a finished `UnifiedTranscript` into a searchable knowledge base — embeds every chunk and writes it into the hybrid vector/full-text Postgres table that `rag_retrieval` later queries. This is where ingestion ends and the retrievable index begins.

## How it does it

`RAGStorageService.index_transcript(source_id, transcript: UnifiedTranscript, release_after=True)` normalizes chunk text, acquires the `RAG_EMBEDDING` model via `ModelManagerOrchestrator`, calls `embed_batch()` (batch size 16, `normalize_embeddings=True`), releases the model if `release_after=True`, and only then calls `replace_rag_chunks()`, which deletes the source's previous rows and inserts the new ones in **one transaction** — re-indexing is replace-in-place, and a failure leaves the old index intact. Returns an `IngestionResult`/`IndexingTelemetry` on success; any failure raises a typed `RAGStorageError` (via `utils/error_mapper.py`) so the job is recorded as FAILED. An empty transcript clears the source's rows.

`HuggingFaceEmbeddingAdapter` (`data/hf_embedder.py`) wraps `sentence_transformers.SentenceTransformer`, running fp16 on CUDA. It wraps the model import in the shared `block_torchcodec_import()` context manager (`app/core/model_manager/utils/torchcodec_guard.py`), a scoped `sys.modules["torchcodec"] = None` restored on exit, to stop torchvision/torchcodec's broken C++ codec bindings from crashing the model load — see Gotchas.

`PostgresRagRepository` persists both a pgvector `embedding` column and an auto-generated `fts_document` (`tsvector`, built with `settings.FTS_LANGUAGE_CONFIG`) computed column on the same table (`VectorChunkModel`/`vector_chunks`) — a single atomic table backs both dense (vector) and sparse (keyword) retrieval; there is no separate vector database.

## Inputs / Outputs

Input: `source_id`, `UnifiedTranscript`. Output: persisted rows in `vector_chunks` (embedding + fts_document + `text_payload` + `raw_text` + `context_text` + `speaker_labels` + timing + source linkage), plus an `IngestionResult`. `text_payload` is the contextualized string that is embedded and keyword-indexed; `raw_text` (the verbatim speaker-labelled transcript) and `context_text` (generated audio/visual context) are stored separately so the answer model is only ever shown source text.

## Where it fits

Phase 16 (`RagIndexingPhase`) — the final ingestion phase, run concurrently with clip extraction. Feeds directly into `rag_retrieval`.

## Gotchas / design notes

The `sys.modules["torchcodec"] = None` scoping hack is *supposed* to be fully contained (restored in `finally`), but `integration_tests/core/model_manager/test_torchcodec_environment_integration.py` proves it can leak across features within one long-lived process — if the RAG embedder loads once, then `torchaudio.load()` runs later in the same process (e.g. a subsequent ingestion job), it can break with `ImportError: TorchCodec is required...`. Relevant to any long-running process (including a future RAG agent) that mixes retrieval and ingestion calls without process isolation.
