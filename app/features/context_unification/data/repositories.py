# Canvas: context_unification_repositories_v1
# Path: app/features/context_unification/data/repositories.py

import uuid
from typing import Optional
from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.context_unification.domain.interfaces import IContextUnificationRepository
from app.features.context_unification.domain.entities import UnifiedTranscript, UnifiedChunk
from app.features.context_unification.data.models import UnifiedTranscriptModel
from app.features.context_unification.utils.error_mapper import map_unification_error


class ContextUnificationRepository(IContextUnificationRepository):
    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, transcript: UnifiedTranscript) -> UnifiedTranscript:
        """
        Persists a newly stitched Unified Transcript to PostgreSQL.
        """
        try:
            # Serialize the complex Pydantic objects to pure dicts for JSONB
            serialized_chunks = [chunk.model_dump(mode='json') for chunk in transcript.chunks]

            db_model = UnifiedTranscriptModel(
                id=transcript.id,
                source_id=transcript.source_id,
                chunks=serialized_chunks,
                created_at=transcript.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return transcript
        except Exception as e:
            raise map_unification_error(e, context=f"Saving Unified Transcript for source {transcript.source_id}")

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[UnifiedTranscript]:
        """
        Retrieves a previously computed Unified Transcript for a given Source.
        """
        try:
            with self._db.session() as session:
                stmt = select(UnifiedTranscriptModel).where(
                    UnifiedTranscriptModel.source_id == source_id
                )
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Deserialization: Rehydrate the raw JSON dictionaries back into strict Pydantic Domain Entities
                domain_chunks = [UnifiedChunk(**c) for c in db_model.chunks]

                return UnifiedTranscript(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    chunks=domain_chunks,
                    created_at=db_model.created_at
                )
        except Exception as e:
            raise map_unification_error(e, context=f"Fetching Unified Transcript by source {source_id}")