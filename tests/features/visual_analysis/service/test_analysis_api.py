# Canvas: test_visual_analysis_api_v1
# Path: tests/features/visual_analysis/service/test_analysis_api.py

import uuid
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from app.core.model_manager.domain.entities import LoadedModel
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames, ExtractedFrame
from app.features.visual_analysis.service.analysis_api import VisualAnalysisAPI
from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis


class TestVisualAnalysisAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "vlm_adapter": MagicMock(),
            "storage_api": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return VisualAnalysisAPI(
            repository=mocks["repo"],
            vlm_adapter=mocks["vlm_adapter"],
            storage_api=mocks["storage_api"]
        )

    def test_process_chunk_idempotent_return(self, api, mocks):
        """If the chunk was already analyzed, return it immediately."""
        chunk = ContextualizedChunk(chunk_id=uuid.uuid4())
        existing_analysis = ChunkVisualAnalysis(source_id=uuid.uuid4(), chunk_id=chunk.chunk_id)

        mocks["repo"].get_by_chunk_id.return_value = existing_analysis

        result = api.process_chunk(
            source_id=uuid.uuid4(),
            chunk=chunk,
            extracted_frames=ExtractedChunkFrames(source_id=uuid.uuid4(), chunk_id=chunk.chunk_id),
            vlm_model=MagicMock(spec=LoadedModel)
        )

        assert result == existing_analysis
        mocks["repo"].get_by_chunk_id.assert_called_once_with(chunk.chunk_id)
        mocks["vlm_adapter"].analyze_frame.assert_not_called()

    def test_process_chunk_empty_frames_saves_empty_ledger(self, api, mocks):
        """If the frame extractor dropped all frames (e.g. blurry), save an empty array."""
        source_id = uuid.uuid4()
        chunk = ContextualizedChunk(chunk_id=uuid.uuid4())
        extracted_frames = ExtractedChunkFrames(source_id=source_id, chunk_id=chunk.chunk_id, frames=[])

        mocks["repo"].get_by_chunk_id.return_value = None
        mocks["repo"].save.side_effect = lambda x: x

        result = api.process_chunk(source_id, chunk, extracted_frames, MagicMock(spec=LoadedModel))

        assert result.chunk_id == chunk.chunk_id
        assert len(result.analyses) == 0
        mocks["repo"].save.assert_called_once()
        mocks["vlm_adapter"].analyze_frame.assert_not_called()

    def test_process_chunk_success_flow(self, api, mocks):
        source_id = uuid.uuid4()
        chunk = ContextualizedChunk(
            chunk_id=uuid.uuid4(),
            original_text="The red line goes up.",
            context_string="Reviewing Q3 performance."
        )

        frame_1 = ExtractedFrame(file_id=uuid.uuid4(), timestamp_sec=1.0, phash_value="abc", quality_score=100)
        frame_2 = ExtractedFrame(file_id=uuid.uuid4(), timestamp_sec=2.0, phash_value="def", quality_score=100)

        extracted_frames = ExtractedChunkFrames(
            source_id=source_id,
            chunk_id=chunk.chunk_id,
            frames=[frame_1, frame_2]
        )

        mock_model = MagicMock(spec=LoadedModel)

        mocks["repo"].get_by_chunk_id.return_value = None
        mocks["repo"].save.side_effect = lambda x: x

        mocks["storage_api"].get_file_path.return_value = Path("/mock/vault/frame.jpg")
        mocks["vlm_adapter"].analyze_frame.side_effect = [
            "A red line going up.",
            "A zoomed in view of the red line."
        ]

        # Execute
        result = api.process_chunk(source_id, chunk, extracted_frames, mock_model)

        # Assertions
        assert result.chunk_id == chunk.chunk_id
        assert len(result.analyses) == 2
        assert result.analyses[0].description == "A red line going up."
        assert result.analyses[1].description == "A zoomed in view of the red line."

        # Verify dependencies were called correctly
        assert mocks["storage_api"].get_file_path.call_count == 2
        assert mocks["vlm_adapter"].analyze_frame.call_count == 2
        mocks["repo"].save.assert_called_once()

    def test_process_chunk_maps_storage_errors(self, api, mocks):
        source_id = uuid.uuid4()
        chunk = ContextualizedChunk(chunk_id=uuid.uuid4())
        frame = ExtractedFrame(file_id=uuid.uuid4(), timestamp_sec=1.0, phash_value="abc", quality_score=100)
        extracted_frames = ExtractedChunkFrames(source_id=source_id, chunk_id=chunk.chunk_id, frames=[frame])

        mocks["repo"].get_by_chunk_id.return_value = None
        mocks["storage_api"].get_file_path.side_effect = Exception("Storage Drive Offline")

        # VisualAnalysisAPI's try/except should catch this and route it through `map_visual_analysis_error`
        with pytest.raises(Exception, match="Unexpected failure in Visual Analysis"):
            api.process_chunk(source_id, chunk, extracted_frames, MagicMock(spec=LoadedModel))