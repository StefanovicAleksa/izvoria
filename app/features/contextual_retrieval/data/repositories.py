# Canvas: contextual_repositories_v1
# Path: app/features/contextual_retrieval/data/repositories.py

import uuid
from typing import Optional

from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.contextual_retrieval.domain.interfaces import IContextualRepository
from app.features.contextual_retrieval.domain.entities import (
    ContextualizedTranscript,
    ContextualizedChunk
)
from app.features.contextual_retrieval.data.models import ContextualizedTranscriptModel
from app.features.contextual_retrieval.utils.error_mapper import map_contextual_error


class ContextualRepository(IContextualRepository):
    """
    PostgreSQL implementation of the Contextual Retrieval Repository.
    Handles the serialization of complex Pydantic aggregates into JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, transcript: ContextualizedTranscript) -> ContextualizedTranscript:
        try:
            # Safely serialize the Pydantic models for Postgres JSONB insertion
            serialized_chunks = [chunk.model_dump(mode='json') for chunk in transcript.chunks]

            db_model = ContextualizedTranscriptModel(
                id=transcript.id,
                source_id=transcript.source_id,
                chunks=serialized_chunks,
                created_at=transcript.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return transcript

        except Exception as e:
            # Context string includes "transcript" to trigger specific database mapping
            raise map_contextual_error(e, context="ContextualRepository.save_transcript") from e

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[ContextualizedTranscript]:
        try:
            with self._db.session() as session:
                stmt = select(ContextualizedTranscriptModel).where(
                    ContextualizedTranscriptModel.source_id == source_id
                )
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Safely deserialize the JSONB back into strong Pydantic Domain Entities
                domain_chunks = [ContextualizedChunk(**c) for c in db_model.chunks]

                return ContextualizedTranscript(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    chunks=domain_chunks,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_contextual_error(e, context="ContextualRepository.get_by_source_id") from e