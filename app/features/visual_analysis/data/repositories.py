# Canvas: visual_analysis_repositories_v1
# Path: app/features/visual_analysis/data/repositories.py

import uuid
from typing import Optional
from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.visual_analysis.domain.interfaces import IVisualAnalysisRepository
from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis, FrameAnalysis
from app.features.visual_analysis.data.models import ChunkVisualAnalysisModel
from app.features.visual_analysis.utils.error_mapper import map_visual_analysis_error


class VisualAnalysisRepository(IVisualAnalysisRepository):
    """
    PostgreSQL implementation of the Visual Analysis Repository.
    Handles the serialization of Pydantic aggregates into JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, visual_analysis: ChunkVisualAnalysis) -> ChunkVisualAnalysis:
        try:
            # Safely serialize the Pydantic models for Postgres JSONB insertion
            serialized_analyses = [analysis.model_dump(mode='json') for analysis in visual_analysis.analyses]

            db_model = ChunkVisualAnalysisModel(
                id=visual_analysis.id,
                source_id=visual_analysis.source_id,
                chunk_id=visual_analysis.chunk_id,
                analyses=serialized_analyses,
                created_at=visual_analysis.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return visual_analysis

        except Exception as e:
            # Context string includes "chunk" to trigger specific database mapping
            raise map_visual_analysis_error(e, context=f"VisualAnalysisRepository.save for chunk {visual_analysis.chunk_id}") from e

    def get_by_chunk_id(self, chunk_id: uuid.UUID) -> Optional[ChunkVisualAnalysis]:
        try:
            with self._db.session() as session:
                stmt = select(ChunkVisualAnalysisModel).where(
                    ChunkVisualAnalysisModel.chunk_id == chunk_id
                )
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Safely deserialize the JSONB back into strong Pydantic Domain Entities
                domain_analyses = [FrameAnalysis(**a) for a in db_model.analyses]

                return ChunkVisualAnalysis(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    chunk_id=db_model.chunk_id,
                    analyses=domain_analyses,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_visual_analysis_error(e, context=f"VisualAnalysisRepository.get_by_chunk_id {chunk_id}") from e