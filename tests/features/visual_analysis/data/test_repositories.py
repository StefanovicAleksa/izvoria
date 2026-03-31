# Canvas: test_visual_analysis_repositories_v1
# Path: tests/features/visual_analysis/data/test_repositories.py

import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.visual_analysis.data.repositories import VisualAnalysisRepository
from app.features.visual_analysis.data.models import ChunkVisualAnalysisModel
from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis, FrameAnalysis
from app.features.visual_analysis.exceptions.database import VisualPersistenceError


class TestVisualAnalysisRepository:
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock()
        mock_session_ctx = MagicMock()
        db_manager.session.return_value = mock_session_ctx
        session = MagicMock()
        mock_session_ctx.__enter__.return_value = session
        return db_manager, session

    @pytest.fixture
    def sample_aggregate(self):
        # A single chunk might have multiple frames if it's long
        frame_1 = FrameAnalysis(frame_id=uuid.uuid4(), description="The doctor points to the fracture.")
        frame_2 = FrameAnalysis(frame_id=uuid.uuid4(), description="The screen shows a 50% bone density drop.")

        return ChunkVisualAnalysis(
            source_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            analyses=[frame_1, frame_2]
        )

    def test_save_happy_path_serializes_jsonb(self, mock_db_manager, sample_aggregate):
        manager, session = mock_db_manager
        repo = VisualAnalysisRepository(manager)

        result = repo.save(sample_aggregate)

        # 1. Verify success return
        assert result == sample_aggregate
        session.add.assert_called_once()

        # 2. Verify SQLAlchemy translation and JSONB nesting
        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, ChunkVisualAnalysisModel)
        assert added_model.chunk_id == sample_aggregate.chunk_id
        assert len(added_model.analyses) == 2

        # Ensure it serialized to raw dicts for Postgres JSONB
        assert added_model.analyses[0]["description"] == "The doctor points to the fracture."
        assert str(sample_aggregate.analyses[0].frame_id) == added_model.analyses[0]["frame_id"]

    def test_get_by_chunk_id_deserializes_jsonb(self, mock_db_manager):
        manager, session = mock_db_manager

        # 1. Setup mock database return
        chunk_id = uuid.uuid4()
        source_id = uuid.uuid4()
        mock_model = MagicMock()
        mock_model.id = uuid.uuid4()
        mock_model.source_id = source_id
        mock_model.chunk_id = chunk_id
        mock_model.created_at = datetime.now(timezone.utc)

        # Simulated raw JSONB from Postgres
        mock_model.analyses = [
            {"frame_id": str(uuid.uuid4()), "description": "Bar chart showing Q3 revenue."}
        ]

        session.execute.return_value.scalar_one_or_none.return_value = mock_model

        # 2. Execute
        repo = VisualAnalysisRepository(manager)
        result = repo.get_by_chunk_id(chunk_id)

        # 3. Assertions
        assert result is not None
        assert result.chunk_id == chunk_id
        assert len(result.analyses) == 1

        # Verify it re-hydrated into strict Domain Pydantic objects
        assert isinstance(result.analyses[0], FrameAnalysis)
        assert result.analyses[0].description == "Bar chart showing Q3 revenue."

    def test_get_by_chunk_id_returns_none_if_missing(self, mock_db_manager):
        manager, session = mock_db_manager
        session.execute.return_value.scalar_one_or_none.return_value = None

        repo = VisualAnalysisRepository(manager)
        result = repo.get_by_chunk_id(uuid.uuid4())

        assert result is None

    def test_save_maps_sqlalchemy_exceptions(self, mock_db_manager, sample_aggregate):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("Database connection dropped during JSONB write.")

        repo = VisualAnalysisRepository(manager)

        with pytest.raises(VisualPersistenceError, match="Database connection dropped"):
            repo.save(sample_aggregate)