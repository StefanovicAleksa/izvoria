# Canvas: test_nlp_router_api_v3
# Path: tests/features/nlp_router/test_router_api.py

import pytest
import uuid
from unittest.mock import MagicMock

from app.features.nlp_router.service.router_api import NLPRouterAPI
from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.features.scene_detector.domain.entities import TimeWindow


class TestNLPRouterAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "orchestrator": MagicMock(),
            "adapter": MagicMock(),
            "manifest": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return NLPRouterAPI(
            orchestrator=mocks["orchestrator"],
            adapter=mocks["adapter"],
            manifest=mocks["manifest"]
        )

    def test_consolidate_windows_logic(self, api):
        """Proves overlapping or adjacent windows are merged, reducing FFmpeg calls."""
        raw_windows = [
            TimeWindow(start_sec=10.0, end_sec=20.0),
            TimeWindow(start_sec=18.0, end_sec=25.0),  # Overlaps previous
            TimeWindow(start_sec=28.0, end_sec=35.0),  # Within 5s gap (28 - 25 = 3) -> Merges
            TimeWindow(start_sec=50.0, end_sec=60.0),  # Large gap -> Does NOT merge
        ]

        result = api._consolidate_windows(raw_windows, gap_threshold_sec=5.0)

        assert len(result) == 2
        assert result[0].start_sec == 10.0
        assert result[0].end_sec == 35.0
        assert result[1].start_sec == 50.0
        assert result[1].end_sec == 60.0

    def test_generate_vision_windows_vram_safety(self, api, mocks):
        """Verifies the Orchestrator safely releases the LLM back to CPU/Disk if a crash occurs."""

        # FIXED: Use a valid TranscriptSegment instead of a MagicMock to pass strict Pydantic rules
        valid_segment = TranscriptSegment(start_sec=0.0, end_sec=10.0, text="Valid text", speaker_id="Speaker")
        dummy_transcript = Transcript(
            source_id=uuid.uuid4(),
            segments=[valid_segment]
        )

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        # Simulate the adapter crashing (e.g., context window overflow)
        mocks["adapter"].extract_visual_cues.side_effect = Exception("CUDA Error")

        with pytest.raises(Exception, match="CUDA Error"):
            api.generate_vision_windows(dummy_transcript)

        # The crash happened, but the finally block MUST still release the VRAM
        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )