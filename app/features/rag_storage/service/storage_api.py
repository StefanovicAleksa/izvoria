# Canvas: rag_storage_api_v3
# Path: app/features/rag_storage/service/storage_api.py

import time
import logging
from typing import List, Optional
from uuid import UUID

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.context_unification.domain.entities import UnifiedTranscript, UnifiedChunk
from app.features.rag_storage.domain.entities import IngestionResult, IndexingTelemetry, StoredChunkText
from app.features.rag_storage.domain.interfaces import IRagRepository, IEmbeddingAdapter
from app.features.rag_storage.utils.error_mapper import map_storage_error
from app.features.rag_storage.utils.payload_formatter import (
    build_payload, normalize_payload_text, source_line, with_source_line
)
from app.features.source_profile.service.source_profile_api import SourceProfileAPI

logger = logging.getLogger(__name__)


class RAGStorageService:
    def __init__(
            self,
            rag_repo: IRagRepository,
            orchestrator: ModelManagerOrchestrator,
            embedder: IEmbeddingAdapter,
            profile_api: Optional[SourceProfileAPI] = None
    ):
        """`profile_api` names each chunk's source in its indexed context; without it chunks are indexed as before."""
        self._rag_repo = rag_repo
        self._orchestrator = orchestrator
        self._embedder = embedder
        self._profile_api = profile_api

    @property
    def _manifest(self) -> ModelManifest:
        return ModelManifest(
            model_type=ModelType.RAG_EMBEDDING,
            model_path=settings.RAG_EMBEDDING_MODEL_PATH,
            required_vram_mb=settings.RAG_EMBEDDING_VRAM_REQUIREMENT_MB
        )

    def _embed(self, texts: List[str], release_after: bool, wait_for_clear: bool = False) -> List[List[float]]:
        loaded_model = self._orchestrator.acquire_model(self._manifest, self._embedder, wait_for_clear=wait_for_clear)
        try:
            return self._embedder.embed_batch(loaded_model, texts)
        finally:
            if release_after:
                self._orchestrator.release_model(loaded_model, self._embedder, flush=True)

    def _source_line(self, source_id: UUID) -> Optional[str]:
        if self._profile_api is None:
            return None
        profile = self._profile_api.get(source_id)
        return source_line(profile.title, profile.topic) if profile else None

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
            try:
                line = self._source_line(source_id)
            except Exception as e:
                # Only costs search a way to find the source by name; the chunks themselves are intact
                logger.warning(f"Indexing source {source_id} without its source line: {e}")
                line = None

            for chunk in chunks:
                if line:
                    chunk = chunk.model_copy(update={
                        "context_text": with_source_line(chunk.context_text or "", line),
                        "text_payload": f"{line}\n{chunk.text_payload}"
                    })
                clean_text = normalize_payload_text(chunk.text_payload)
                normalized_texts.append(clean_text)
                normalized_chunks.append(chunk.model_copy(update={"text_payload": clean_text}))

            embeddings = self._embed(normalized_texts, release_after)

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

    def stored_texts(self, source_id: UUID) -> List[str]:
        """A source's indexed raw texts in reading order: what it says, without any generated context."""
        try:
            return [chunk.raw_text for chunk in self._rag_repo.read_chunk_texts(source_id) if chunk.raw_text]
        except Exception as e:
            raise map_storage_error(e, context=f"Reading the stored texts of source {source_id}") from e

    def refresh_source_line(self, source_id: UUID, release_after: bool = True) -> int:
        """
        Re-indexes an already indexed source under its current profile: each chunk's `[Source: …]` line is replaced
        and the chunk re-embedded in place, with the same chunk ids. Nothing is re-read from the recording or
        re-contextualised, so this takes the embedder alone (seconds per source). Returns the chunks updated; 0 when
        the source has no profile or no chunks. It waits for the GPU rather than evicting a model in use, since a
        person renaming a source is a live call.
        """
        try:
            line = self._source_line(source_id)
            if not line:
                return 0
            stored = self._rag_repo.read_chunk_texts(source_id)
            if not stored:
                return 0

            updated = [
                StoredChunkText(chunk_id=chunk.chunk_id, raw_text=chunk.raw_text,
                                context_text=with_source_line(chunk.context_text or "", line))
                for chunk in stored
            ]
            payloads = [normalize_payload_text(build_payload(chunk.context_text, chunk.raw_text)) for chunk in updated]
            embeddings = self._embed(payloads, release_after, wait_for_clear=True)
            return self._rag_repo.update_chunk_payloads(source_id, updated, payloads, embeddings)

        except Exception as e:
            logger.error(f"Failed to refresh the source line of source {source_id}: {e}", exc_info=True)
            raise map_storage_error(e, context=f"Refreshing the source line of source {source_id}") from e
