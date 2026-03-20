# Canvas: test_correction_repository_v2
# Path: tests/features/asr_error_correction/test_repository.py

import uuid
from datetime import datetime, timezone
import pytest
from unittest.mock import MagicMock

from app.legacy_features.asr_error_correction.domain.entities import DocumentGlossary, DomainCorrectedTranscript
from app.legacy_features.asr_error_correction.data.repositories import CorrectionRepository
from app.legacy_features.asr_error_correction.data.models import DocumentGlossaryModel, DomainCorrectedTranscriptModel


class TestCorrectionRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    def test_save_glossary_success(self, mock_db_manager):
        manager, session = mock_db_manager
        repo = CorrectionRepository(manager)

        glossary = DocumentGlossary(source_id=uuid.uuid4(), domain_category="Legal", items=[])
        result = repo.save_glossary(glossary)

        assert result == glossary
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, DocumentGlossaryModel)
        assert added_model.source_id == glossary.source_id

    def test_get_glossary_found(self, mock_db_manager):
        manager, session = mock_db_manager
        repo = CorrectionRepository(manager)

        source_id = uuid.uuid4()
        # FIX: Manually inject created_at so Pydantic validation passes without a real DB flush
        mock_db_model = DocumentGlossaryModel(
            id=uuid.uuid4(),
            source_id=source_id,
            domain_category="Legal",
            items=[],
            created_at=datetime.now(timezone.utc)
        )
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        result = repo.get_glossary(source_id)

        assert result is not None
        assert result.source_id == source_id

    def test_save_corrected_transcript_success(self, mock_db_manager):
        manager, session = mock_db_manager
        repo = CorrectionRepository(manager)

        transcript = DomainCorrectedTranscript(source_id=uuid.uuid4(), sentences=[], audit_trail=[])
        result = repo.save_corrected_transcript(transcript)

        assert result == transcript
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, DomainCorrectedTranscriptModel)

    def test_get_corrected_transcript_found(self, mock_db_manager):
        manager, session = mock_db_manager
        repo = CorrectionRepository(manager)

        source_id = uuid.uuid4()
        # FIX: Manually inject created_at so Pydantic validation passes without a real DB flush
        mock_db_model = DomainCorrectedTranscriptModel(
            id=uuid.uuid4(),
            source_id=source_id,
            sentences=[],
            audit_trail=[],
            created_at=datetime.now(timezone.utc)
        )
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        result = repo.get_corrected_transcript(source_id)

        assert result is not None
        assert result.source_id == source_id