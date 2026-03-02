# Canvas: scene_detector_repositories_v1
# Path: app/features/scene_detector/data/repositories.py

import uuid
from typing import Optional
from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.scene_detector.domain.interfaces import ISceneRepository
from app.features.scene_detector.domain.entities import SceneMap, DetectedScene
from app.features.scene_detector.data.models import SceneMapModel
from app.features.scene_detector.utils.error_mapper import map_scene_detector_error


class SceneRepository(ISceneRepository):
    """
    PostgreSQL implementation of the Scene Repository using SQLAlchemy and JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save_map(self, scene_map: SceneMap) -> SceneMap:
        try:
            # Serialize the Pydantic Domain models into raw dictionaries for the JSONB column
            serialized_scenes = [scene.model_dump(mode='json') for scene in scene_map.scenes]

            db_model = SceneMapModel(
                id=scene_map.id,
                source_id=scene_map.source_id,
                scenes=serialized_scenes,
                created_at=scene_map.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return scene_map

        except Exception as e:
            # Safely route the SQL error (e.g., duplicate source_id) to the Domain exception
            raise map_scene_detector_error(e) from e

    def get_map_by_source_id(self, source_id: uuid.UUID) -> Optional[SceneMap]:
        try:
            with self._db.session() as session:
                stmt = select(SceneMapModel).where(SceneMapModel.source_id == source_id)
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Safely deserialize the JSONB back into strictly-typed Domain entities
                domain_scenes = [DetectedScene(**scene) for scene in db_model.scenes]

                return SceneMap(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    scenes=domain_scenes,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_scene_detector_error(e) from e