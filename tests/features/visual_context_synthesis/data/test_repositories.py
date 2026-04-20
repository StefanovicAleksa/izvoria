# Canvas: test_visual_synthesis_repositories_v1
# Path: tests/features/visual_context_synthesis/data/test_repositories.py

import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.visual_context_synthesis.data.repositories import VisualSynthesisRepository
from app.features.visual_context_synthesis.data.models import VisualTimelineModel
from app.features.visual_context_synthesis.domain.entities import VisualTimeline, VisualTimelineChunk, \
    VisualContextFrame
from app.features.visual_context_synthesis.exceptions.database import VisualTimelinePersistenceError


class TestVisualSynthesisRepository:
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock()
        mock_session_ctx = MagicMock()
        db_manager.session.return_value = mock_session_ctx
        session = MagicMock()
        mock_session_ctx.__enter__.return_value = session
        return db_manager, session

    @pytest.fixture
    def sample_timeline(self):
        frame = VisualContextFrame(frame_id=uuid.uuid4(), timestamp_sec=10.5)
        chunk = VisualTimelineChunk(
            chunk_id=uuid.uuid4(),
            visual_context_string="A line chart showing a spike.",
            visual_evidence=[frame]
        )
        return VisualTimeline(
            source_id=uuid.uuid4(),
            visual_chunks=[chunk]
        )

    def test_save_happy_path_serializes_jsonb(self, mock_db_manager, sample_timeline):
        manager, session = mock_db_manager
        repo = VisualSynthesisRepository(manager)

        result = repo.save(sample_timeline)

        assert result == sample_timeline
        session.add.assert_called_once()

        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, VisualTimelineModel)
        assert added_model.source_id == sample_timeline.source_id
        assert len(added_model.visual_chunks) == 1

        # Verify JSONB serialization
        assert added_model.visual_chunks[0]["visual_context_string"] == "A line chart showing a spike."
        assert added_model.visual_chunks[0]["visual_evidence"][0]["timestamp_sec"] == 10.5

    def test_get_by_source_id_deserializes_jsonb(self, mock_db_manager):
        manager, session = mock_db_manager

        source_id = uuid.uuid4()
        mock_model = MagicMock()
        mock_model.id = uuid.uuid4()
        mock_model.source_id = source_id
        mock_model.created_at = datetime.now(timezone.utc)

        # Simulated JSONB from Postgres
        mock_model.visual_chunks = [{
            "chunk_id": str(uuid.uuid4()),
            "visual_context_string": "A bar chart.",
            "visual_evidence": [{"frame_id": str(uuid.uuid4()), "timestamp_sec": 5.0}]
        }]

        session.execute.return_value.scalar_one_or_none.return_value = mock_model

        repo = VisualSynthesisRepository(manager)
        result = repo.get_by_source_id(source_id)

        assert result is not None
        assert result.source_id == source_id
        assert len(result.visual_chunks) == 1
        assert isinstance(result.visual_chunks[0], VisualTimelineChunk)
        assert result.visual_chunks[0].visual_context_string == "A bar chart."
        assert isinstance(result.visual_chunks[0].visual_evidence[0], VisualContextFrame)

    def test_save_maps_sqlalchemy_exceptions(self, mock_db_manager, sample_timeline):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB transaction failed.")

        repo = VisualSynthesisRepository(manager)

        with pytest.raises(VisualTimelinePersistenceError):
            repo.save(sample_timeline)