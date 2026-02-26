# Canvas: test_repository_v2
# Path: tests/features/diarization/test_repository.py

import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.diarization.domain.entities import DiarizationMap, DiarizationSegment
from app.features.diarization.data.models import DiarizationMapModel
from app.features.diarization.data.repositories import DiarizationRepository
from app.features.diarization.exceptions.database import DiarizationDatabaseError


class TestDiarizationRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        # Mock the context manager behavior: with db.session() as session:
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    @pytest.fixture
    def sample_map(self):
        return DiarizationMap(
            source_id=uuid.uuid4(),
            segments=[DiarizationSegment(start_sec=1.5, end_sec=3.0, speaker_id="Speaker_0")]
        )

    def test_save_map_success(self, mock_db_manager, sample_map):
        manager, session = mock_db_manager
        repo = DiarizationRepository(manager)

        result = repo.save_map(sample_map)

        assert result == sample_map

        # Ensure the model is queued for insertion
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, DiarizationMapModel)

        # Removed the session.commit() assertion because DatabaseManager automatically handles it
        # upon exiting the 'with self._db.session():' context block.

    def test_get_map_by_source_id_found(self, mock_db_manager, sample_map):
        manager, session = mock_db_manager

        mock_db_model = DiarizationMapModel(
            id=sample_map.id,
            source_id=sample_map.source_id,
            segments=[seg.model_dump() for seg in sample_map.segments],
            created_at=sample_map.created_at
        )
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        repo = DiarizationRepository(manager)
        result = repo.get_map_by_source_id(sample_map.source_id)

        assert result is not None
        assert result.source_id == sample_map.source_id
        assert len(result.segments) == 1
        assert result.segments[0].speaker_id == "Speaker_0"

    def test_repository_maps_db_exceptions(self, mock_db_manager, sample_map):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB Connection Lost")

        repo = DiarizationRepository(manager)

        with pytest.raises(DiarizationDatabaseError):
            repo.save_map(sample_map)