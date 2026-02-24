# filename: test_vad_repository_v2.py
# path: tests/features/vad/test_repository.py
import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.vad.data.models import VadMapModel
from app.features.vad.data.repositories import VadRepository
from app.features.vad.exceptions.database import VadDatabaseError


class TestVadRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    @pytest.fixture
    def sample_vad_map(self):
        source_id = uuid.uuid4()
        segments = [VadSegment(start_sec=1.0, end_sec=2.5, confidence=0.9)]
        return VadMap(source_id=source_id, segments=segments)

    def test_save_map_success(self, mock_db_manager, sample_vad_map):
        manager, session = mock_db_manager
        repo = VadRepository(manager)

        result = repo.save_map(sample_vad_map)

        assert result == sample_vad_map
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, VadMapModel)
        assert added_model.source_id == sample_vad_map.source_id
        assert added_model.segments[0]["start_sec"] == 1.0

    def test_get_map_by_source_id_found(self, mock_db_manager, sample_vad_map):
        manager, session = mock_db_manager

        # Mock the DB returning a result INSIDE the context manager
        mock_db_model = VadMapModel(
            id=sample_vad_map.id,
            source_id=sample_vad_map.source_id,
            segments=[seg.model_dump() for seg in sample_vad_map.segments],
            created_at=sample_vad_map.created_at
        )
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        repo = VadRepository(manager)
        result = repo.get_map_by_source_id(sample_vad_map.source_id)

        assert result is not None
        assert result.source_id == sample_vad_map.source_id
        assert len(result.segments) == 1
        assert result.segments[0].start_sec == 1.0

    def test_get_map_by_source_id_not_found(self, mock_db_manager):
        manager, session = mock_db_manager
        session.execute.return_value.scalar_one_or_none.return_value = None

        repo = VadRepository(manager)
        result = repo.get_map_by_source_id(uuid.uuid4())

        assert result is None

    def test_repository_maps_db_exceptions(self, mock_db_manager, sample_vad_map):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB Connection Lost")

        repo = VadRepository(manager)

        with pytest.raises(VadDatabaseError):
            repo.save_map(sample_vad_map)