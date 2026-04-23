# Canvas: rag_storage_postgres_repo_v1
# Path: app/features/rag_storage/data/postgres_dense_repo.py

import logging
from typing import List
from uuid import UUID
from sqlalchemy import delete

from app.core.database.service.manager import db_manager
from app.features.rag_storage.domain.interfaces import IDenseRepository
from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_storage.data.models import VectorChunkModel
from app.features.rag_storage.utils.error_mapper import map_storage_error

logger = logging.getLogger(__name__)


class PostgresDenseRepository(IDenseRepository):
    """
    Concrete implementation of IDenseRepository using PostgreSQL and pgvector.
    """

    def save_dense_vectors(
            self,
            source_id: UUID,
            chunks: List[UnifiedChunk],
            embeddings: List[List[float]]
    ) -> int:
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError("Mismatch between number of chunks and embeddings.")

        try:
            with db_manager.session() as session:
                # Prepare bulk insert objects
                vector_models = []
                for chunk, embedding in zip(chunks, embeddings):
                    # We assume chunk.text_payload holds the fully unified string
                    model = VectorChunkModel(
                        source_id=source_id,
                        chunk_id=chunk.chunk_id,
                        start_sec=chunk.start_sec,
                        end_sec=chunk.end_sec,
                        text_payload=chunk.text_payload,
                        embedding=embedding
                    )
                    vector_models.append(model)

                session.bulk_save_objects(vector_models)
                session.commit()

                logger.info(f"Successfully saved {len(vector_models)} dense vectors for source {source_id}.")
                return len(vector_models)

        except Exception as e:
            raise map_storage_error(e, context=f"Saving dense vectors for source {source_id}")

    def delete_by_source_id(self, source_id: UUID) -> None:
        try:
            with db_manager.session() as session:
                stmt = delete(VectorChunkModel).where(VectorChunkModel.source_id == source_id)
                result = session.execute(stmt)
                session.commit()

                if result.rowcount > 0:
                    logger.info(f"Cleared {result.rowcount} existing vectors for source {source_id}.")
        except Exception as e:
            raise map_storage_error(e, context=f"Clearing vectors for source {source_id}")