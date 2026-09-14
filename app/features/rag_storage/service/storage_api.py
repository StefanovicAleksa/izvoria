# Canvas: rag_storage_api_v3
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
from app.features.rag_storage.domain.interfaces import IRagRepository, IEmbeddingAdapter
from app.features.rag_storage.utils.error_mapper import map_storage_error
from app.features.rag_storage.utils.payload_formatter import normalize_payload_text

logger = logging.getLogger(__name__)


class RAGStorageService:
    def __init__(
            self,
            rag_repo: IRagRepository,
            orchestrator: ModelManagerOrchestrator,
            embedder: IEmbeddingAdapter
    ):
        self._rag_repo = rag_repo
        self._orchestrator = orchestrator
        self._embedder = embedder

    def index_transcript(self, source_id: UUID, transcript: UnifiedTranscript,
                         release_after: bool = True) -> IngestionResult:
        """
        Embeds and indexes a Unified Transcript. Failures raise a typed RAGStorageError so the
        job is recorded as FAILED; previously indexed chunks stay intact until the new set is
        fully embedded and swapped in by a single transaction.
        """
        start_time = time.time()
        chunks = transcript.chunks

        try:
            if not chunks:
                self._rag_repo.delete_by_source_id(source_id)
                return IngestionResult(success=True, telemetry=IndexingTelemetry(source_id, 0, 0, 0, 0.0, "none"))

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

            # Single atomic swap of both Dense and Sparse data, only after embedding succeeded
            chunks_saved = self._rag_repo.replace_rag_chunks(source_id, normalized_chunks, embeddings)

            duration = round(time.time() - start_time, 2)
            return IngestionResult(
                success=True,
                telemetry=IndexingTelemetry(source_id, len(normalized_chunks), chunks_saved, chunks_saved, duration,
                                           settings.RAG_EMBEDDING_MODEL_PATH)
            )

        except Exception as e:
            logger.error(f"Failed to index transcript for source {source_id}: {e}", exc_info=True)
            raise map_storage_error(e, context=f"Indexing transcript for source {source_id}") from e
