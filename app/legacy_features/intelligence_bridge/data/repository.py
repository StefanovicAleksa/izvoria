import uuid
from typing import Optional
from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.legacy_features.intelligence_bridge.domain.interfaces import IIntelligentSceneRepository
from app.legacy_features.intelligence_bridge.domain.entities import IntelligentSceneMap, EnrichedScene
from app.legacy_features.intelligence_bridge.data.models import SceneMapModel
from app.legacy_features.intelligence_bridge.utils.error_mapper import map_bridge_error


class IntelligentSceneRepository(IIntelligentSceneRepository):
    """
    PostgreSQL implementation using the familiar SceneMap model names and JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save_map(self, scene_map: IntelligentSceneMap) -> IntelligentSceneMap:
        try:
            # Serialize the Pydantic Domain models into raw dictionaries for the JSONB column
            serialized_scenes = [scene.model_dump(mode='json') for scene in scene_map.scenes]

            with self._db.session() as session:
                # 1. Idempotency: Wipe existing if re-running the pipeline
                stmt = select(SceneMapModel).where(SceneMapModel.source_id == scene_map.source_id)
                existing = session.execute(stmt).scalar_one_or_none()

                if existing:
                    session.delete(existing)
                    session.flush()

                # 2. Insert the new enriched map
                db_model = SceneMapModel(
                    id=scene_map.id,
                    source_id=scene_map.source_id,
                    scenes=serialized_scenes,
                    created_at=scene_map.created_at
                )
                session.add(db_model)

            return scene_map

        except Exception as e:
            raise map_bridge_error(e) from e

    def get_map_by_source_id(self, source_id: uuid.UUID) -> Optional[IntelligentSceneMap]:
        try:
            with self._db.session() as session:
                stmt = select(SceneMapModel).where(SceneMapModel.source_id == source_id)
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Deserialize back to the EnrichedScene domain entities
                domain_scenes = [EnrichedScene(**scene) for scene in db_model.scenes]

                return IntelligentSceneMap(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    scenes=domain_scenes,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_bridge_error(e) from e