# filename: vad_repositories_v4.py
# path: app/features/vad/data/repositories.py
import uuid
from typing import Optional

from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.vad.domain.interfaces import IVadRepository
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.vad.data.models import VadMapModel
from app.features.vad.utils.error_mapper import map_vad_error


class VadRepository(IVadRepository):
    """
    PostgreSQL implementation of the VAD Repository using SQLAlchemy and JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save_map(self, vad_map: VadMap) -> VadMap:
        try:
            serialized_segments = [segment.model_dump() for segment in vad_map.segments]

            db_model = VadMapModel(
                id=vad_map.id,
                source_id=vad_map.source_id,
                segments=serialized_segments,
                created_at=vad_map.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return vad_map

        except Exception as e:
            raise map_vad_error(e) from e

    def get_map_by_source_id(self, source_id: uuid.UUID) -> Optional[VadMap]:
        try:
            with self._db.session() as session:
                stmt = select(VadMapModel).where(VadMapModel.source_id == source_id)
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # CRITICAL FIX: Deserialize the JSONB while the session is STILL OPEN
                domain_segments = [VadSegment(**seg) for seg in db_model.segments]

                return VadMap(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    segments=domain_segments,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_vad_error(e) from e