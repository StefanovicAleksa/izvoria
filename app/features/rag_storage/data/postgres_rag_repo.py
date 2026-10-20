# Canvas: rag_storage_postgres_repo_v5
# Path: app/features/rag_storage/data/postgres_rag_repo.py

import logging
from typing import List, Optional
from uuid import UUID
from sqlalchemy import delete, select, update

from app.core.config import settings
from app.core.database.domain.interfaces import IDatabaseManager
from app.core.database.service.manager import db_manager
from app.features.rag_storage.domain.interfaces import IRagRepository
from app.features.rag_storage.domain.entities import StoredChunkText
from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_storage.data.models import VectorChunkModel
from app.features.rag_storage.utils.error_mapper import map_storage_error

logger = logging.getLogger(__name__)


class PostgresRagRepository(IRagRepository):
    """
    Unified implementation of IRagRepository.
    Leverages PostgreSQL for both pgvector (Dense) and TSVECTOR FTS (Sparse) indexing.
    """

    def __init__(self, database: Optional[IDatabaseManager] = None):
        """`database` is a firm's database on the hosted API; without it, the appliance's own (`db_manager`)."""
        self._database = database

    @property
    def _db(self) -> IDatabaseManager:
        return self._database or db_manager

    def replace_rag_chunks(
            self,
            source_id: UUID,
            chunks: List[UnifiedChunk],
            embeddings: List[List[float]]
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Mismatch between number of chunks and embeddings.")

        try:
            # One session == one transaction: the delete only commits if the insert succeeds.
            with self._db.session() as session:
                deleted = session.execute(delete(VectorChunkModel).where(VectorChunkModel.source_id == source_id))
                if deleted.rowcount:
                    logger.info(f"Replacing {deleted.rowcount} existing RAG chunks for source {source_id}.")

                # A batch is ingested in the active language; its chunks are keyword-indexed with that stemmer
                fts_config = settings.FTS_LANGUAGE_CONFIG
                vector_models = [
                    VectorChunkModel(
                        source_id=source_id,
                        chunk_id=chunk.chunk_id,
                        anchor_kind=chunk.anchor_kind.value,
                        start_sec=chunk.start_sec,
                        end_sec=chunk.end_sec,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section_path=chunk.section_path,
                        # Chunks arrive in reading order (a transcript's timeline, a document's pages)
                        position=position,
                        text_payload=chunk.text_payload,
                        # Chunks built outside context unification may carry no separate raw text;
                        # their payload is then the only text available.
                        raw_text=chunk.raw_text or chunk.text_payload,
                        context_text=chunk.context_text or None,
                        speaker_labels=list(chunk.speaker_labels),
                        embedding=embedding,
                        fts_config=fts_config
                    )
                    for position, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ]

                # bulk_save_objects safely ignores the Computed fts_document column
                # and lets Postgres generate it upon insertion.
                session.bulk_save_objects(vector_models)

                logger.info(f"Successfully saved {len(vector_models)} hybrid chunks for source {source_id}.")
                return len(vector_models)

        except Exception as e:
            raise map_storage_error(e, context=f"Replacing hybrid RAG chunks for source {source_id}") from e

    def delete_by_source_id(self, source_id: UUID) -> None:
        try:
            with self._db.session() as session:
                # This single delete now removes the vectors AND the full-text search index
                stmt = delete(VectorChunkModel).where(VectorChunkModel.source_id == source_id)
                result = session.execute(stmt)

                if result.rowcount > 0:
                    logger.info(f"Cleared {result.rowcount} existing RAG chunks for source {source_id}.")
        except Exception as e:
            raise map_storage_error(e, context=f"Clearing RAG chunks for source {source_id}") from e

    def read_chunk_texts(self, source_id: UUID) -> List[StoredChunkText]:
        try:
            with self._db.session() as session:
                rows = session.execute(
                    select(VectorChunkModel.chunk_id, VectorChunkModel.context_text, VectorChunkModel.raw_text)
                    .where(VectorChunkModel.source_id == source_id)
                    .order_by(VectorChunkModel.position, VectorChunkModel.start_sec)
                ).all()
                return [StoredChunkText(chunk_id=row.chunk_id, context_text=row.context_text, raw_text=row.raw_text)
                        for row in rows]
        except Exception as e:
            raise map_storage_error(e, context=f"Reading stored chunk texts for source {source_id}") from e

    def update_chunk_payloads(self, source_id: UUID, chunks: List[StoredChunkText], payloads: List[str],
                              embeddings: List[List[float]]) -> int:
        if not len(chunks) == len(payloads) == len(embeddings):
            raise ValueError("Mismatch between chunks, payloads and embeddings.")
        try:
            # One transaction: every chunk of the source moves to its new payload, or none does. fts_document is
            # generated from text_payload, so Postgres re-tokenizes it with the chunk's own fts_config.
            with self._db.session() as session:
                updated = 0
                for chunk, payload, embedding in zip(chunks, payloads, embeddings):
                    result = session.execute(
                        update(VectorChunkModel)
                        .where(VectorChunkModel.source_id == source_id, VectorChunkModel.chunk_id == chunk.chunk_id)
                        .values(context_text=chunk.context_text or None, text_payload=payload, embedding=embedding)
                    )
                    updated += result.rowcount
                return updated
        except Exception as e:
            raise map_storage_error(e, context=f"Updating chunk payloads for source {source_id}") from e
