# Path: app/features/rag_storage/service/storage_api.py

import time
import logging
from typing import List
from uuid import UUID

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.context_unification.domain.entities import UnifiedTranscript, UnifiedChunk
from app.features.rag_storage.domain.entities import IngestionResult, IndexingTelemetry
from app.features.rag_storage.domain.interfaces import IDenseRepository, ISparseRepository, IEmbeddingAdapter
from app.features.rag_storage.utils.payload_formatter import normalize_payload_text

logger = logging.getLogger(__name__)


class RAGStorageService:
    def __init__(
            self,
            dense_repo: IDenseRepository,
            sparse_repo: ISparseRepository,
            orchestrator: ModelManagerOrchestrator,
            embedder: IEmbeddingAdapter
    ):
        self._dense_repo = dense_repo
        self._sparse_repo = sparse_repo
        self._orchestrator = orchestrator
        self._embedder = embedder

    def index_transcript(self, source_id: UUID, transcript: UnifiedTranscript,
                         release_after: bool = True) -> IngestionResult:
        start_time = time.time()
        chunks = transcript.chunks

        if not chunks:
            return IngestionResult(success=True, telemetry=IndexingTelemetry(source_id, 0, 0, 0, 0.0, "none"))

        try:
            self._dense_repo.delete_by_source_id(source_id)
            self._sparse_repo.delete_by_source_id(source_id)

            normalized_texts: List[str] = []
            normalized_chunks: List[UnifiedChunk] = []

            for chunk in chunks:
                clean_text = normalize_payload_text(chunk.text_payload)
                normalized_texts.append(clean_text)
                normalized_chunks.append(chunk.model_copy(update={"text_payload": clean_text}))

            manifest = ModelManifest(
                model_type=ModelType.RAG_EMBEDDING,
                model_path=settings.RAG_EMBEDDING_MODEL_PATH,
                required_vram_mb=settings.RAG_EMBEDDING_VRAM_REQUIREMENT_MB
            )

            loaded_model = self._orchestrator.acquire_model(manifest, self._embedder)
            try:
                embeddings = self._embedder.embed_batch(loaded_model, normalized_texts)
            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._embedder, flush=True)

            dense_saved = self._dense_repo.save_dense_vectors(source_id, normalized_chunks, embeddings)
            sparse_saved = self._sparse_repo.save_sparse_index(source_id, normalized_chunks)

            duration = round(time.time() - start_time, 2)
            return IngestionResult(
                success=True,
                telemetry=IndexingTelemetry(source_id, len(normalized_chunks), dense_saved, sparse_saved, duration,
                                            settings.RAG_EMBEDDING_MODEL_PATH)
            )

        except Exception as e:
            logger.error(f"Failed to index transcript: {e}", exc_info=True)
            return IngestionResult(
                success=False,
                telemetry=IndexingTelemetry(source_id, len(chunks), 0, 0, round(time.time() - start_time, 2), "failed"),
                error_message=f"Internal Server Error: {str(e)}"
            )