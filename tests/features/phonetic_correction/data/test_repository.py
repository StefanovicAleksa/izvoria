import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.phonetic_correction.domain.entities import PhoneticCorrectedTranscript
from app.features.phonetic_correction.data.models import PhoneticCorrectedTranscriptModel
from app.features.phonetic_correction.data.repositories import PhoneticRepository
from app.features.phonetic_correction.exceptions.database import TranscriptSaveError


class TestPhoneticRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    def test_save_success(self, mock_db_manager):
        manager, session = mock_db_manager
        repo = PhoneticRepository(manager)

        transcript = PhoneticCorrectedTranscript(source_id=uuid.uuid4(), sentences=[], audit_trail=[])
        result = repo.save(transcript)

        assert result == transcript
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, PhoneticCorrectedTranscriptModel)

    def test_db_exceptions_mapped(self, mock_db_manager):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB Down")
        repo = PhoneticRepository(manager)

        transcript = PhoneticCorrectedTranscript(source_id=uuid.uuid4(), sentences=[], audit_trail=[])
        with pytest.raises(TranscriptSaveError):
            repo.save(transcript)