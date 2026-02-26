# Path: app/features/diarization/data/repositories.py

import uuid
from typing import Optional

from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.diarization.domain.interfaces import IDiarizationRepository
from app.features.diarization.domain.entities import DiarizationMap, DiarizationSegment
from app.features.diarization.data.models import DiarizationMapModel
from app.features.diarization.utils.error_mapper import map_diarization_error


class DiarizationRepository(IDiarizationRepository):
    """
    PostgreSQL implementation of the Diarization Repository using SQLAlchemy and JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save_map(self, diarization_map: DiarizationMap) -> DiarizationMap:
        try:
            serialized_segments = [segment.model_dump() for segment in diarization_map.segments]

            db_model = DiarizationMapModel(
                id=diarization_map.id,
                source_id=diarization_map.source_id,
                segments=serialized_segments,
                created_at=diarization_map.created_at
            )

            with self._db.session() as session:
                session.add(db_model)
                session.commit()  # <--- FIXED: Explicitly commit the transaction to Postgres

            return diarization_map

        except Exception as e:
            raise map_diarization_error(e) from e

    def get_map_by_source_id(self, source_id: uuid.UUID) -> Optional[DiarizationMap]:
        try:
            with self._db.session() as session:
                stmt = select(DiarizationMapModel).where(DiarizationMapModel.source_id == source_id)
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                domain_segments = [DiarizationSegment(**seg) for seg in db_model.segments]

                return DiarizationMap(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    segments=domain_segments,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_diarization_error(e) from e