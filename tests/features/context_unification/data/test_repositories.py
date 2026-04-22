# Canvas: test_context_unification_repositories_v1
# Path: tests/features/context_unification/data/test_repositories.py

import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.context_unification.data.repositories import ContextUnificationRepository
from app.features.context_unification.data.models import UnifiedTranscriptModel
from app.features.context_unification.domain.entities import UnifiedTranscript, UnifiedChunk
from app.features.visual_context_synthesis.domain.entities import VisualContextFrame
from app.features.context_unification.exceptions.database import UnifiedTranscriptPersistenceError


class TestContextUnificationRepository:
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock()
        mock_session_ctx = MagicMock()
        db_manager.session.return_value = mock_session_ctx
        session = MagicMock()
        mock_session_ctx.__enter__.return_value = session
        return db_manager, session

    @pytest.fixture
    def sample_transcript(self):
        frame = VisualContextFrame(frame_id=uuid.uuid4(), timestamp_sec=1.0)
        chunk = UnifiedChunk(
            chunk_id=uuid.uuid4(),
            start_sec=0.0,
            end_sec=5.0,
            text_payload="[Audio] \n [Visual] \n Text",
            visual_evidence=[frame]
        )
        return UnifiedTranscript(
            source_id=uuid.uuid4(),
            chunks=[chunk]
        )

    def test_save_happy_path_serializes_jsonb(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager
        repo = ContextUnificationRepository(manager)

        result = repo.save(sample_transcript)

        assert result == sample_transcript
        session.add.assert_called_once()

        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, UnifiedTranscriptModel)
        assert added_model.source_id == sample_transcript.source_id
        assert len(added_model.chunks) == 1
        assert added_model.chunks[0]["text_payload"] == "[Audio] \n [Visual] \n Text"

    def test_get_by_source_id_deserializes_jsonb(self, mock_db_manager):
        manager, session = mock_db_manager
        source_id = uuid.uuid4()

        mock_model = MagicMock()
        mock_model.id = uuid.uuid4()
        mock_model.source_id = source_id
        mock_model.created_at = datetime.now(timezone.utc)
        mock_model.chunks = [{
            "chunk_id": str(uuid.uuid4()),
            "start_sec": 0.0,
            "end_sec": 10.0,
            "text_payload": "Rehydrated Payload",
            "visual_evidence": [{"frame_id": str(uuid.uuid4()), "timestamp_sec": 5.0}]
        }]

        session.execute.return_value.scalar_one_or_none.return_value = mock_model

        repo = ContextUnificationRepository(manager)
        result = repo.get_by_source_id(source_id)

        assert result is not None
        assert result.source_id == source_id
        assert len(result.chunks) == 1
        assert isinstance(result.chunks[0], UnifiedChunk)
        assert result.chunks[0].text_payload == "Rehydrated Payload"

    def test_save_maps_sqlalchemy_exceptions(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB transaction failed.")

        repo = ContextUnificationRepository(manager)

        with pytest.raises(UnifiedTranscriptPersistenceError):
            repo.save(sample_transcript)