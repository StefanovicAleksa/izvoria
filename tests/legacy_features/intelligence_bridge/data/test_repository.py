# Canvas: test_repository_v1
# Path: tests/features/intelligence_bridge/test_repository.py

import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.legacy_features.intelligence_bridge.domain.entities import IntelligentSceneMap, EnrichedScene
from app.legacy_features.intelligence_bridge.data.models import SceneMapModel
from app.legacy_features.intelligence_bridge.data.repository import IntelligentSceneRepository
from app.legacy_features.intelligence_bridge.exceptions.sync import BridgeDatabaseError


class TestIntelligentSceneRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    @pytest.fixture
    def sample_map(self):
        source_id = uuid.uuid4()
        scenes = [EnrichedScene(
            timestamp_sec=10.0,
            extracted_file_id=uuid.uuid4(),
            phash_value="abc",
            extraction_method="global_pass",
            semantic_context=None
        )]
        return IntelligentSceneMap(source_id=source_id, scenes=scenes)

    def test_save_map_inserts_and_flushes(self, mock_db_manager, sample_map):
        manager, session = mock_db_manager

        # [cite_start]Simulate no existing map to delete [cite: 1099]
        session.execute.return_value.scalar_one_or_none.return_value = None

        repo = IntelligentSceneRepository(manager)
        result = repo.save_map(sample_map)

        assert result == sample_map
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]

        assert isinstance(added_model, SceneMapModel)
        assert added_model.source_id == sample_map.source_id
        assert added_model.scenes[0]["extraction_method"] == "global_pass"

    def test_get_map_by_source_id_found(self, mock_db_manager, sample_map):
        manager, session = mock_db_manager

        mock_db_model = SceneMapModel(
            id=sample_map.id,
            source_id=sample_map.source_id,
            scenes=[scene.model_dump(mode='json') for scene in sample_map.scenes],
            created_at=sample_map.created_at
        )
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        repo = IntelligentSceneRepository(manager)
        result = repo.get_map_by_source_id(sample_map.source_id)

        assert result is not None
        assert result.source_id == sample_map.source_id
        assert len(result.scenes) == 1
        assert result.scenes[0].timestamp_sec == 10.0

    def test_repository_maps_db_exceptions(self, mock_db_manager, sample_map):
        manager, session = mock_db_manager
        session.execute.side_effect = SQLAlchemyError("DB failure")

        repo = IntelligentSceneRepository(manager)

        with pytest.raises(BridgeDatabaseError):
            repo.save_map(sample_map)