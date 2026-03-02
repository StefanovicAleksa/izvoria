# Canvas: test_repository_v1.py
# Path: tests/features/scene_detector/test_repository.py

import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.scene_detector.domain.entities import SceneMap, DetectedScene
from app.features.scene_detector.data.models import SceneMapModel
from app.features.scene_detector.data.repositories import SceneRepository
from app.features.scene_detector.exceptions.database import SceneDetectorDatabaseError


class TestSceneRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    @pytest.fixture
    def sample_scene_map(self):
        source_id = uuid.uuid4()
        scenes = [DetectedScene(timestamp_sec=12.5, extracted_file_id=uuid.uuid4(), phash_value="abc")]
        return SceneMap(source_id=source_id, scenes=scenes)

    def test_save_map_success(self, mock_db_manager, sample_scene_map):
        manager, session = mock_db_manager
        repo = SceneRepository(manager)

        result = repo.save_map(sample_scene_map)

        assert result == sample_scene_map
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]

        assert isinstance(added_model, SceneMapModel)
        assert added_model.source_id == sample_scene_map.source_id
        assert added_model.scenes[0]["timestamp_sec"] == 12.5

    def test_get_map_by_source_id_found(self, mock_db_manager, sample_scene_map):
        manager, session = mock_db_manager

        mock_db_model = SceneMapModel(
            id=sample_scene_map.id,
            source_id=sample_scene_map.source_id,
            scenes=[scene.model_dump() for scene in sample_scene_map.scenes],
            created_at=sample_scene_map.created_at
        )
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        repo = SceneRepository(manager)
        result = repo.get_map_by_source_id(sample_scene_map.source_id)

        assert result is not None
        assert result.source_id == sample_scene_map.source_id
        assert len(result.scenes) == 1
        assert result.scenes[0].timestamp_sec == 12.5

    def test_repository_maps_db_exceptions(self, mock_db_manager, sample_scene_map):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB failure")

        repo = SceneRepository(manager)

        with pytest.raises(SceneDetectorDatabaseError):
            repo.save_map(sample_scene_map)