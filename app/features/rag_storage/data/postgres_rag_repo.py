# Canvas: rag_storage_postgres_repo_v5
# Path: app/features/rag_storage/data/postgres_rag_repo.py

import logging
from typing import List
from uuid import UUID
from sqlalchemy import delete

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
            with db_manager.session() as session:
                deleted = session.execute(delete(VectorChunkModel).where(VectorChunkModel.source_id == source_id))
                if deleted.rowcount:
                    logger.info(f"Replacing {deleted.rowcount} existing RAG chunks for source {source_id}.")

                vector_models = [
                    VectorChunkModel(
                        source_id=source_id,
                        chunk_id=chunk.chunk_id,
                        start_sec=chunk.start_sec,
                        end_sec=chunk.end_sec,
                        text_payload=chunk.text_payload,
                        # Chunks built outside context unification may carry no separate raw text;
                        # their payload is then the only text available.
                        raw_text=chunk.raw_text or chunk.text_payload,
                        context_text=chunk.context_text or None,
                        speaker_labels=list(chunk.speaker_labels),
                        embedding=embedding
                    )
                    for chunk, embedding in zip(chunks, embeddings)
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
            with db_manager.session() as session:
                # This single delete now removes the vectors AND the full-text search index
                stmt = delete(VectorChunkModel).where(VectorChunkModel.source_id == source_id)
                result = session.execute(stmt)

                if result.rowcount > 0:
                    logger.info(f"Cleared {result.rowcount} existing RAG chunks for source {source_id}.")
        except Exception as e:
            raise map_storage_error(e, context=f"Clearing RAG chunks for source {source_id}") from e
