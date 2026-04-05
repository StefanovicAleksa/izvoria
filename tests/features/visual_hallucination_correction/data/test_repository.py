# Canvas: test_hallucination_repositories_v1
# Path: tests/features/visual_hallucination_correction/data/test_repository.py

import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.visual_hallucination_correction.domain.entities import (
    VisualHallucinationEvaluatedTranscript,
    VisualHallucinationAudit
)
from app.features.visual_hallucination_correction.data.models import VisualHallucinationEvaluationModel
from app.features.visual_hallucination_correction.data.repositories import VisualHallucinationRepository
from app.features.visual_hallucination_correction.exceptions.database import AuditPersistenceError


class TestVisualHallucinationRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    def test_save_success(self, mock_db_manager):
        manager, session = mock_db_manager
        repo = VisualHallucinationRepository(manager)

        audit = VisualHallucinationAudit(
            chunk_id=uuid.uuid4(),
            is_visual_accepted=False,
            discard_reason="Irrelevant background"
        )
        transcript = VisualHallucinationEvaluatedTranscript(source_id=uuid.uuid4(), audits=[audit])

        result = repo.save(transcript)

        assert result == transcript
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]

        assert isinstance(added_model, VisualHallucinationEvaluationModel)
        assert len(added_model.audits) == 1
        assert added_model.audits[0]["is_visual_accepted"] is False

    def test_get_by_source_id_found(self, mock_db_manager):
        manager, session = mock_db_manager
        repo = VisualHallucinationRepository(manager)

        mock_db_model = MagicMock()
        mock_db_model.id = uuid.uuid4()
        mock_db_model.source_id = uuid.uuid4()
        mock_db_model.audits = [{
            "chunk_id": str(uuid.uuid4()),
            "is_visual_accepted": True,
            "discard_reason": None
        }]
        mock_db_model.created_at = datetime.now(timezone.utc)

        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        result = repo.get_by_source_id(mock_db_model.source_id)

        assert result is not None
        assert result.source_id == mock_db_model.source_id
        assert len(result.audits) == 1
        assert result.audits[0].is_visual_accepted is True

    def test_db_exceptions_mapped(self, mock_db_manager):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB Down")
        repo = VisualHallucinationRepository(manager)

        transcript = VisualHallucinationEvaluatedTranscript(source_id=uuid.uuid4(), audits=[])

        with pytest.raises(AuditPersistenceError):
            repo.save(transcript)