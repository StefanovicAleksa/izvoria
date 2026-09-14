# Path: app/features/rag_reranking/service/rerank_api.py

import os
import logging
from typing import List

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.rag_reranking.domain.interfaces import IRerankerAdapter
from app.features.rag_reranking.exceptions.model import RerankInferenceError
from app.features.rag_reranking.utils.error_mapper import map_rerank_error

logger = logging.getLogger(__name__)


class RerankAPI:
    """
    Second-stage relevance ranking for hybrid search results (the reranking step of Anthropic's
    Contextual Retrieval). A cross-encoder reads the query and each candidate chunk together,
    which ranks far more precisely than the RRF fusion of independent dense/sparse scores.
    """

    def __init__(self, orchestrator: ModelManagerOrchestrator, adapter: IRerankerAdapter):
        self._orchestrator = orchestrator
        self._adapter = adapter
        self._manifest = ModelManifest(
            model_type=ModelType.RAG_RERANKING,
            model_path=settings.RAG_RERANKER_MODEL_PATH,
            required_vram_mb=settings.RAG_RERANKER_VRAM_REQUIREMENT_MB
        )

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    def is_available(self) -> bool:
        """The reranker is optional: without weights on disk, results keep their RRF order."""
        return os.path.exists(self._manifest.model_path)

    def rerank(
            self,
            query: str,
            chunks: List[RetrievedChunk],
            top_n: int,
            release_after: bool = True
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []

        if not self.is_available():
            logger.warning(
                f"Reranker weights not found at {self._manifest.model_path}. "
                f"Returning the top {top_n} chunks in RRF order."
            )
            return chunks[:top_n]

        try:
            loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter, wait_for_clear=True)
            try:
                scores = self._adapter.score(loaded_model, query, [chunk.text_payload for chunk in chunks])
            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

            if len(scores) != len(chunks):
                raise RerankInferenceError(
                    f"Reranker returned {len(scores)} scores for {len(chunks)} chunks."
                )

            # sorted() is stable: equal scores keep their RRF order
            ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
            reranked = [chunk.model_copy(update={"rerank_score": float(score)}) for chunk, score in ranked[:top_n]]

            logger.info(f"Reranked {len(chunks)} candidates down to {len(reranked)}.")
            return reranked

        except Exception as e:
            raise map_rerank_error(e, context=f"Reranking {len(chunks)} chunks for query '{query}'") from e
