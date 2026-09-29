# Canvas: rag_retrieval_service_v3
# Path: app/features/rag_retrieval/service/retrieval_api.py

import time
import uuid
import logging
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.language.service.language_api import get_supported_language_packs
from app.core.language.utils.script import normalize_script
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.rag_retrieval.domain.entities import (
    BatchSearchResult, CorpusSweep, IndexedSource, RetrievedChunk, SearchQuery, SearchResult,
)
from app.features.rag_retrieval.domain.interfaces import IRetrievalRepository, IPromptEmbeddingAdapter
from app.features.rag_retrieval.utils.error_mapper import map_retrieval_error

logger = logging.getLogger(__name__)

class RetrievalAPI:
    """
    The entry point for executing hybrid RAG searches.
    Coordinates VRAM model loading, text vectorization, and Postgres query execution.
    """
    def __init__(
        self,
        orchestrator: ModelManagerOrchestrator,
        embedding_adapter: IPromptEmbeddingAdapter,
        search_repo: IRetrievalRepository
    ):
        self._orchestrator = orchestrator
        self._embedding_adapter = embedding_adapter
        self._search_repo = search_repo

    def search(self, query: SearchQuery) -> SearchResult:
        """
        Returns the RRF-fused candidate pool (at least `RAG_RETRIEVAL_CANDIDATES` chunks, ranked).
        Narrowing the pool down to `query.limit` is the reranker's job, so relevant chunks that
        fuse just below the final cut still get a chance to be scored.
        """
        candidate_count = max(query.limit, settings.RAG_RETRIEVAL_CANDIDATES)
        logger.info(f"Executing RAG Retrieval for query: '{query.prompt}' (Candidates: {candidate_count})")
        start_time = time.time()

        try:
            # 1-2. VRAM Orchestration: Load -> Embed -> Unload (Zero-Leakage Pattern)
            loaded_model = self._orchestrator.acquire_model(
                self._embedding_manifest(), self._embedding_adapter, wait_for_clear=True
            )
            try:
                query_vector = self._embedding_adapter.embed_prompt(loaded_model, _search_text(query.prompt))
            finally:
                # CRITICAL: Always release immediately to free VRAM for the imminent LLM inference
                self._orchestrator.release_model(loaded_model, self._embedding_adapter, flush=True)

            # 3. Execute the Hybrid Postgres Search
            retrieved_chunks = self._search_repo.execute_hybrid_search(
                query_vector=query_vector,
                query_text=_search_text(query.prompt),
                limit=candidate_count,
                source_id_filter=query.source_id_filter,
                # The pool goes to the reranker: the dense leg's best must be in it (see postgres_search_repo)
                dense_keep=settings.RAG_DENSE_CANDIDATES_KEPT
            )

            # 4. Format Final Payload
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks in {execution_time_ms}ms.")

            return SearchResult(
                query=query.prompt,
                results=retrieved_chunks,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            raise map_retrieval_error(e, context=f"Processing search query: '{query.prompt}'") from e

    def _embedding_manifest(self) -> ModelManifest:
        return ModelManifest(
            model_type=ModelType.RAG_EMBEDDING,
            model_path=settings.RAG_EMBEDDING_MODEL_PATH,
            required_vram_mb=settings.RAG_EMBEDDING_VRAM_REQUIREMENT_MB
        )

    def list_indexed_sources(self, min_chunks: int = 1) -> List[IndexedSource]:
        """What is actually retrievable, as opposed to what was ingested. Needs no model."""
        try:
            return self._search_repo.list_indexed_sources(min_chunks)
        except Exception as e:
            raise map_retrieval_error(e, context="Listing indexed sources") from e

    def search_batch(
        self,
        queries: List[SearchQuery],
        sweep: Optional[CorpusSweep] = None,
        candidates: Optional[int] = None,
    ) -> BatchSearchResult:
        """
        Runs a whole fan-out -- and optionally a per-document sweep -- under ONE embedder acquisition.

        This is the method the report mode's model-swap budget is built on. `search()` loads the embedder,
        embeds one string, and unloads; doing that forty times costs forty loads of a 2.2 GB model, which on
        a single-GPU box is minutes of pure loading. Here the model is acquired once, every distinct prompt
        is embedded in one forward pass, and the SQL then runs with no model resident at all.

        Distinct prompts are embedded once and their vectors reused, which matters because every one of the
        sweep's per-source queries shares the topic's vector.
        """
        if not queries and sweep is None:
            return BatchSearchResult()

        start_time = time.time()
        candidate_count = candidates or settings.RAG_RETRIEVAL_CANDIDATES

        try:
            prompts = [query.prompt for query in queries]
            if sweep is not None:
                prompts.append(sweep.prompt)

            # Distinct, order-preserving: repeated prompts cost one forward pass, not one each.
            unique_prompts = list(dict.fromkeys(prompts))

            loaded_model = self._orchestrator.acquire_model(
                self._embedding_manifest(), self._embedding_adapter, wait_for_clear=True
            )
            try:
                vectors = self._embedding_adapter.embed_prompts(
                    loaded_model, [_search_text(prompt) for prompt in unique_prompts]
                )
            finally:
                # Released before any SQL runs: the database work needs no VRAM, and the next stage's
                # model should not have to wait on an embedder that is only holding the slot.
                self._orchestrator.release_model(loaded_model, self._embedding_adapter, flush=True)

            vector_by_prompt: Dict[str, List[float]] = dict(zip(unique_prompts, vectors))
            logger.info(
                f"Embedded {len(unique_prompts)} distinct prompt(s) for {len(queries)} quer(ies)"
                f"{' plus a corpus sweep' if sweep is not None else ''} in one model acquisition."
            )

            results = [
                SearchResult(
                    query=query.prompt,
                    results=self._search_repo.execute_hybrid_search(
                        query_vector=vector_by_prompt[query.prompt],
                        query_text=_search_text(query.prompt),
                        limit=max(query.limit, candidate_count),
                        source_id_filter=query.source_id_filter,
                        dense_keep=settings.RAG_DENSE_CANDIDATES_KEPT,
                    ),
                    execution_time_ms=0,
                )
                for query in queries
            ]

            sweep_results: List[RetrievedChunk] = []
            swept_source_count = 0
            if sweep is not None:
                sweep_results, swept_source_count = self._run_sweep(
                    sweep, vector_by_prompt[sweep.prompt]
                )

            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Batch search returned {sum(len(r.results) for r in results)} fan-out chunk(s) and "
                f"{len(sweep_results)} sweep chunk(s) from {swept_source_count} source(s) "
                f"in {execution_time_ms}ms."
            )

            return BatchSearchResult(
                results=results,
                sweep_results=sweep_results,
                swept_source_count=swept_source_count,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            raise map_retrieval_error(e, context=f"Executing a batch of {len(queries)} search(es)") from e

    def _run_sweep(self, sweep: CorpusSweep, query_vector: List[float]) -> tuple:
        """
        The per-document sweep: pick the best-matching sources, then search inside each one.

        Each picked source gets a full hybrid (dense + sparse) search via the existing search SQL, rather
        than being served straight from the ANN probe that picked it. The probe only decides *where* to
        look; letting it also decide *what* to return would throw away the keyword half of the retrieval
        on exactly the sources the fan-out already failed to surface.
        """
        source_ids = self._search_repo.pick_sweep_sources(
            query_vector=query_vector,
            ann_depth=sweep.ann_depth,
            max_sources=sweep.max_sources,
        )
        if not source_ids:
            return [], 0

        chunks: List[RetrievedChunk] = []
        for source_id in source_ids:
            chunks.extend(self._search_repo.execute_hybrid_search(
                query_vector=query_vector,
                query_text=_search_text(sweep.prompt),
                limit=sweep.per_source_limit,
                source_id_filter=source_id,
            ))

        return chunks, len(source_ids)

    def get_chunks(self, chunk_ids: List[uuid.UUID]) -> List[RetrievedChunk]:
        """Chunks by id, in the given order; deleted ones are skipped. Needs no model."""
        try:
            return self._search_repo.fetch_chunks(chunk_ids)
        except Exception as e:
            raise map_retrieval_error(e, context="Fetching chunks by id") from e

    def expand_neighbors(self, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """
        Appends the chunks on either side of the `top_k` best chunks (small-to-big retrieval): a transcript
        answer often starts in one chunk and finishes in the next. Neighbours go after every given chunk, so
        a context budget drops them first. Needs no model.
        """
        if top_k <= 0 or not chunks:
            return list(chunks)

        try:
            anchors = [chunk.chunk_id for chunk in chunks[:top_k]]
            present = {chunk.chunk_id for chunk in chunks}
            neighbors = [n for n in self._search_repo.fetch_neighbors(anchors) if n.chunk_id not in present]
            neighbors.sort(key=lambda n: (str(n.source_id), n.position, n.start_sec))

            if neighbors:
                logger.info(f"Added {len(neighbors)} neighbouring chunks around the top {len(anchors)} results.")
            return list(chunks) + neighbors

        except Exception as e:
            raise map_retrieval_error(e, context="Expanding results with neighbouring chunks") from e


def _search_text(prompt: str) -> str:
    """
    The text actually embedded and keyword-matched: the prompt in the index's scripts. The index holds every
    language, so every pack's script rule applies whatever language is active: a Serbian query typed in Cyrillic
    searches the Latin Serbian chunks even while English is active. Results stay keyed by the prompt as written.
    """
    for pack in get_supported_language_packs():
        prompt = normalize_script(prompt, pack.transcription.script)
    return prompt
