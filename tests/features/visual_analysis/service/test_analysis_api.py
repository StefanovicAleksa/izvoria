# Canvas: test_visual_analysis_api_v2
# Path: tests/features/visual_analysis/service/test_analysis_api.py

import uuid
import pytest
from unittest.mock import MagicMock, call, patch
from pathlib import Path

from app.core.model_manager.domain.entities import LoadedModel
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames, ExtractedFrame
from app.features.visual_analysis.service.analysis_api import VisualAnalysisAPI
from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis, FrameObservation

MODULE = "app.features.visual_analysis.service.analysis_api"


def _frame(timestamp_sec: float) -> ExtractedFrame:
    return ExtractedFrame(file_id=uuid.uuid4(), timestamp_sec=timestamp_sec, phash_value="abc", quality_score=100)


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
        mocks["vlm_adapter"].analyze_frames.assert_not_called()

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
        mocks["vlm_adapter"].analyze_frames.assert_not_called()

    def test_process_chunk_describes_frames_in_one_call(self, api, mocks):
        source_id = uuid.uuid4()
        chunk = ContextualizedChunk(
            chunk_id=uuid.uuid4(),
            original_text="The red line goes up.",
            context_string="Reviewing Q3 performance."
        )
        frame_1, frame_2 = _frame(1.0), _frame(2.0)
        extracted_frames = ExtractedChunkFrames(source_id=source_id, chunk_id=chunk.chunk_id, frames=[frame_1, frame_2])
        mock_model = MagicMock(spec=LoadedModel)

        mocks["repo"].get_by_chunk_id.return_value = None
        mocks["repo"].save.side_effect = lambda x: x
        mocks["storage_api"].get_file_path.side_effect = [Path("/vault/a.jpg"), Path("/vault/b.jpg")]
        mocks["vlm_adapter"].analyze_frames.return_value = [
            FrameObservation(shows="line chart", legible_text=["Q3 Revenue"], marks="red line going up",
                             speech_link="the red line"),
            FrameObservation(shows="zoomed line chart", legible_text=["Q3 Revenue", "+4%"]),
        ]

        result = api.process_chunk(source_id, chunk, extracted_frames, mock_model)

        mocks["vlm_adapter"].analyze_frames.assert_called_once_with(
            model=mock_model,
            image_paths=[Path("/vault/a.jpg"), Path("/vault/b.jpg")],
            original_text="The red line goes up.",
            context_string="Reviewing Q3 performance."
        )
        assert mocks["storage_api"].get_file_path.call_args_list == [call(frame_1.file_id), call(frame_2.file_id)]
        # Rendered from the fields; text the first frame already reported isn't repeated for the second
        assert [(a.frame_id, a.description, a.timestamp_sec) for a in result.analyses] == [
            (frame_1.frame_id, 'Line chart with on-screen text "Q3 Revenue". Marked on screen: red line going up. '
                               "Speaker's reference: the red line.", 1.0),
            (frame_2.frame_id, 'Zoomed line chart with on-screen text "+4%".', 2.0),
        ]
        mocks["repo"].save.assert_called_once()

    @patch(f"{MODULE}.settings")
    def test_process_chunk_groups_frames_chronologically_per_call(self, mock_settings, api, mocks):
        mock_settings.VISION_MAX_IMAGES_PER_CALL = 2
        source_id = uuid.uuid4()
        chunk = ContextualizedChunk(chunk_id=uuid.uuid4())
        late, early, middle = _frame(3.0), _frame(1.0), _frame(2.0)
        extracted_frames = ExtractedChunkFrames(source_id=source_id, chunk_id=chunk.chunk_id, frames=[late, early, middle])

        mocks["repo"].get_by_chunk_id.return_value = None
        mocks["repo"].save.side_effect = lambda x: x
        mocks["storage_api"].get_file_path.side_effect = lambda file_id: Path(f"/vault/{file_id}.jpg")
        mocks["vlm_adapter"].analyze_frames.side_effect = [
            [FrameObservation(shows="slide", legible_text=["first"]), FrameObservation(shows="slide", legible_text=["second"])],
            [FrameObservation(shows="slide", legible_text=["third"])],
        ]

        result = api.process_chunk(source_id, chunk, extracted_frames, MagicMock(spec=LoadedModel))

        calls = mocks["vlm_adapter"].analyze_frames.call_args_list
        assert [c.kwargs["image_paths"] for c in calls] == [
            [Path(f"/vault/{early.file_id}.jpg"), Path(f"/vault/{middle.file_id}.jpg")],
            [Path(f"/vault/{late.file_id}.jpg")],
        ]
        assert [(a.frame_id, a.description) for a in result.analyses] == [
            (early.frame_id, 'Slide with on-screen text "first".'),
            (middle.frame_id, 'Slide with on-screen text "second".'),
            (late.frame_id, 'Slide with on-screen text "third".'),
        ]

    def test_process_chunk_stores_no_description_for_frames_without_visible_evidence(self, api, mocks):
        source_id = uuid.uuid4()
        chunk = ContextualizedChunk(chunk_id=uuid.uuid4(), original_text="The level is 142.27.")
        frames = [_frame(1.0), _frame(2.0), _frame(3.0), _frame(4.0), _frame(5.0)]
        extracted_frames = ExtractedChunkFrames(source_id=source_id, chunk_id=chunk.chunk_id, frames=frames)

        mocks["repo"].get_by_chunk_id.return_value = None
        mocks["repo"].save.side_effect = lambda x: x
        mocks["storage_api"].get_file_path.return_value = Path("/vault/frame.jpg")
        mocks["vlm_adapter"].analyze_frames.side_effect = [
            [
                FrameObservation(shows="slide", legible_text=["Exhibit A"]),
                None,                                                          # unstructured answer
                FrameObservation(shows="speaker only"),                        # talking head
                FrameObservation(shows="chart", legible_text=["142.27 level"]),  # copied from the speech
            ],
            [FrameObservation(shows="slide", legible_text=["exhibit a"])],     # repeats an earlier frame
        ]

        result = api.process_chunk(source_id, chunk, extracted_frames, MagicMock(spec=LoadedModel))

        assert [(a.frame_id, a.description) for a in result.analyses] == [
            (frames[0].frame_id, 'Slide with on-screen text "Exhibit A".')
        ]
        mocks["repo"].save.assert_called_once_with(result)

    def test_process_chunk_maps_storage_errors(self, api, mocks):
        source_id = uuid.uuid4()
        chunk = ContextualizedChunk(chunk_id=uuid.uuid4())
        extracted_frames = ExtractedChunkFrames(source_id=source_id, chunk_id=chunk.chunk_id, frames=[_frame(1.0)])

        mocks["repo"].get_by_chunk_id.return_value = None
        mocks["storage_api"].get_file_path.side_effect = Exception("Storage Drive Offline")

        # VisualAnalysisAPI's try/except should catch this and route it through `map_visual_analysis_error`
        with pytest.raises(Exception, match="Unexpected failure in Visual Analysis"):
            api.process_chunk(source_id, chunk, extracted_frames, MagicMock(spec=LoadedModel))


class TestDescribeImage:
    def test_grounds_the_observation_without_a_transcript(self):
        adapter = MagicMock()
        adapter.describe_image.return_value = FrameObservation(
            shows="A photo of a street sign", legible_text=["Main St", "www.example.com", "main st"], marks=None)
        api = VisualAnalysisAPI(repository=MagicMock(), vlm_adapter=adapter, storage_api=MagicMock())
        model = MagicMock(spec=LoadedModel)

        result = api.describe_image(Path("/vault/sign.jpg"), model)

        adapter.describe_image.assert_called_once_with(model, Path("/vault/sign.jpg"))
        assert result == FrameObservation(shows="A photo of a street sign", legible_text=["Main St"])

    def test_nothing_structured_is_none(self):
        adapter = MagicMock()
        adapter.describe_image.return_value = None
        api = VisualAnalysisAPI(repository=MagicMock(), vlm_adapter=adapter, storage_api=MagicMock())

        assert api.describe_image(Path("/vault/x.jpg"), MagicMock(spec=LoadedModel)) is None
