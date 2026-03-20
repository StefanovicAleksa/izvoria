# filename: tests/features/audio_extractor/test_repository.py

import pytest
import uuid
from unittest.mock import Mock, MagicMock

from app.core.database import exceptions as db_exc
from app.features.audio_extractor.data.repositories import AudioExtractionRepository
from app.features.audio_extractor.domain.entities import AudioExtraction
from app.features.audio_extractor.exceptions.base import AudioExtractionDatabaseError


class TestAudioExtractionRepository:
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock()
        mock_session_ctx = MagicMock()
        db_manager.session.return_value = mock_session_ctx
        session = MagicMock()
        mock_session_ctx.__enter__.return_value = session
        return db_manager

    @pytest.fixture
    def repository(self, mock_db_manager):
        return AudioExtractionRepository(mock_db_manager)

    def test_create_happy_path(self, repository, mock_db_manager):
        """Verifies an extraction record is properly added to the DB session."""
        extraction = AudioExtraction(
            id=uuid.uuid4(),
            source_id=uuid.uuid4(),
            extracted_file_id=uuid.uuid4()
        )

        result = repository.create(extraction)

        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.assert_called_once()

        added_model = session.add.call_args[0][0]
        assert added_model.id == extraction.id
        assert added_model.source_id == extraction.source_id
        assert result == extraction

    def test_create_db_error(self, repository, mock_db_manager):
        """Verifies DB errors are correctly wrapped."""
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.side_effect = db_exc.DatabaseError("DB connection failed")

        extraction = AudioExtraction(id=uuid.uuid4(), source_id=uuid.uuid4(), extracted_file_id=uuid.uuid4())

        with pytest.raises(AudioExtractionDatabaseError, match="DB connection failed"):
            repository.create(extraction)

    def test_get_latest_by_source_id_found(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value

        mock_model = MagicMock()
        mock_model.id = uuid.uuid4()
        mock_model.source_id = uuid.uuid4()
        mock_model.extracted_file_id = uuid.uuid4()

        session.execute.return_value.scalar_one_or_none.return_value = mock_model

        result = repository.get_latest_by_source_id(mock_model.source_id)

        assert result is not None
        assert result.id == mock_model.id
        assert result.source_id == mock_model.source_id