# Canvas: rag_retrieval_service_v3
# Path: app/features/rag_retrieval/service/retrieval_api.py

import time
import logging
from typing import List

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
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
            # 1. Define the model requirements
            manifest = ModelManifest(
                model_type=ModelType.RAG_EMBEDDING,
                model_path=settings.RAG_EMBEDDING_MODEL_PATH,
                required_vram_mb=settings.RAG_EMBEDDING_VRAM_REQUIREMENT_MB
            )

            # 2. VRAM Orchestration: Load -> Embed -> Unload (Zero-Leakage Pattern)
            loaded_model = self._orchestrator.acquire_model(manifest, self._embedding_adapter, wait_for_clear=True)
            try:
                query_vector = self._embedding_adapter.embed_prompt(loaded_model, query.prompt)
            finally:
                # CRITICAL: Always release immediately to free VRAM for the imminent LLM inference
                self._orchestrator.release_model(loaded_model, self._embedding_adapter, flush=True)

            # 3. Execute the Hybrid Postgres Search
            retrieved_chunks = self._search_repo.execute_hybrid_search(
                query_vector=query_vector,
                query_text=query.prompt,
                limit=candidate_count,
                source_id_filter=query.source_id_filter
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
            neighbors.sort(key=lambda n: (str(n.source_id), n.start_sec))

            if neighbors:
                logger.info(f"Added {len(neighbors)} neighbouring chunks around the top {len(anchors)} results.")
            return list(chunks) + neighbors

        except Exception as e:
            raise map_retrieval_error(e, context="Expanding results with neighbouring chunks") from e
