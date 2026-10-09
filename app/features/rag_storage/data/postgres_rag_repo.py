# Canvas: rag_storage_postgres_repo_v5
# Path: app/features/rag_storage/data/postgres_rag_repo.py

import logging
from typing import List, Optional
from uuid import UUID
from sqlalchemy import delete

from app.core.config import settings
from app.core.database.domain.interfaces import IDatabaseManager
from app.core.database.service.manager import db_manager
from app.features.rag_storage.domain.interfaces import IRagRepository
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
