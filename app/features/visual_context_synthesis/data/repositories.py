# Canvas: visual_synthesis_repository_v1
# Path: app/features/visual_context_synthesis/data/repositories.py

import uuid
from typing import Optional
from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.visual_context_synthesis.domain.interfaces import IVisualSynthesisRepository
from app.features.visual_context_synthesis.domain.entities import VisualTimeline, VisualTimelineChunk
from app.features.visual_context_synthesis.data.models import VisualTimelineModel
from app.features.visual_context_synthesis.utils.error_mapper import map_visual_synthesis_error


class VisualSynthesisRepository(IVisualSynthesisRepository):
    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, timeline: VisualTimeline) -> VisualTimeline:
        """
        Persists a synthesized visual timeline to PostgreSQL.
        """
        try:
            # Serialize domain objects to JSON-compatible dicts for JSONB storage
            serialized_chunks = [chunk.model_dump(mode='json') for chunk in timeline.visual_chunks]

            db_model = VisualTimelineModel(
                id=timeline.id,
                source_id=timeline.source_id,
                visual_chunks=serialized_chunks,
                created_at=timeline.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return timeline
        except Exception as e:
            raise map_visual_synthesis_error(e, context=f"Saving visual timeline for source {timeline.source_id}")

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[VisualTimeline]:
        """
        Retrieves a visual timeline by its associated Source ID.
        """
        try:
            with self._db.session() as session:
                stmt = select(VisualTimelineModel).where(
                    VisualTimelineModel.source_id == source_id
                )
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Reconstruct Domain Entities from JSONB data
                domain_chunks = [VisualTimelineChunk(**c) for c in db_model.visual_chunks]

                return VisualTimeline(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    visual_chunks=domain_chunks,
                    created_at=db_model.created_at
                )
        except Exception as e:
            raise map_visual_synthesis_error(e, context=f"Fetching visual timeline for source {source_id}")