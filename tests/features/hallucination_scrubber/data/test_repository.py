# Canvas: test_repository_v1
# Path: tests/features/hallucination_scrubber/data/test_repository.py

import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.hallucination_scrubber.domain.entities import HallucinationScrubbedTranscript
from app.features.hallucination_scrubber.data.models import HallucinationScrubbedTranscriptModel
from app.features.hallucination_scrubber.data.repositories import HallucinationRepository
from app.features.hallucination_scrubber.exceptions.database import ScrubbedTranscriptSaveError


class TestHallucinationRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    def test_save_success(self, mock_db_manager):
        manager, session = mock_db_manager
        repo = HallucinationRepository(manager)

        transcript = HallucinationScrubbedTranscript(source_id=uuid.uuid4(), segments=[], audit_trail=[])
        result = repo.save(transcript)

        assert result == transcript
        session.merge.assert_called_once()
        merged_model = session.merge.call_args[0][0]
        assert isinstance(merged_model, HallucinationScrubbedTranscriptModel)
        assert merged_model.source_id == transcript.source_id

    def test_db_exceptions_mapped_to_domain_errors(self, mock_db_manager):
        manager, session = mock_db_manager
        session.merge.side_effect = SQLAlchemyError("Postgres connection dropped")
        repo = HallucinationRepository(manager)

        transcript = HallucinationScrubbedTranscript(source_id=uuid.uuid4(), segments=[], audit_trail=[])

        with pytest.raises(ScrubbedTranscriptSaveError):
            repo.save(transcript)