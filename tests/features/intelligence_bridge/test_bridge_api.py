# Canvas: test_bridge_api_v3
# Path: tests/features/intelligence_bridge/test_bridge_api.py

import pytest
import uuid
from unittest.mock import MagicMock

from app.features.intelligence_bridge.service.bridge_api import IntelligenceBridgeAPI
from app.features.intelligence_bridge.domain.entities import IntelligentSceneMap
from app.features.scene_detector.domain.entities import SceneMap, DetectedScene, TimeWindow
from app.features.transcription.domain.entities import Transcript, TranscriptSegment


class TestIntelligenceBridgeAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "scene_api": MagicMock(),
            "transcription_api": MagicMock(),
            "nlp_router_api": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return IntelligenceBridgeAPI(
            repository=mocks["repo"],
            scene_api=mocks["scene_api"],
            transcription_api=mocks["transcription_api"],
            nlp_router_api=mocks["nlp_router_api"]
        )

    def test_process_multimodal_timeline_idempotency(self, api, mocks):
        """Verifies instant return if the IntelligentSceneMap already exists."""
        source_id = uuid.uuid4()
        existing_map = IntelligentSceneMap(source_id=source_id, scenes=[])
        mocks["repo"].get_map_by_source_id.return_value = existing_map

        result = api.process_multimodal_timeline(source_id, "vid.mp4")

        assert result == existing_map
        mocks["scene_api"].process_video.assert_not_called()

    def test_fallback_to_global_no_transcript(self, api, mocks):
        """Verifies pipeline gracefully stops and saves global pass if audio fails/empty."""
        source_id = uuid.uuid4()
        mocks["repo"].get_map_by_source_id.return_value = None

        global_map = SceneMap(source_id=source_id, scenes=[
            DetectedScene(timestamp_sec=5.0, extracted_file_id=uuid.uuid4(), phash_value="abc")
        ])
        mocks["scene_api"].process_video.return_value = global_map
        mocks["transcription_api"].get_transcript.return_value = None
        mocks["repo"].save_map.side_effect = lambda x: x

        result = api.process_multimodal_timeline(source_id, "vid.mp4")

        assert isinstance(result, IntelligentSceneMap)
        assert len(result.scenes) == 1
        assert result.scenes[0].extraction_method == "global_pass"
        mocks["nlp_router_api"].generate_vision_windows.assert_not_called()

    def test_full_pass_success(self, api, mocks):
        """Verifies full 4-pass pipeline execution."""
        source_id = uuid.uuid4()
        mocks["repo"].get_map_by_source_id.return_value = None

        # Pass 1: Global
        global_map = SceneMap(source_id=source_id, scenes=[
            DetectedScene(timestamp_sec=5.0, extracted_file_id=uuid.uuid4(), phash_value="111")
        ])
        # Pass 3: Semantic
        semantic_map = SceneMap(source_id=source_id, scenes=[
            DetectedScene(timestamp_sec=15.0, extracted_file_id=uuid.uuid4(), phash_value="fff")
        ])

        mocks["scene_api"].process_video.side_effect = [global_map, semantic_map]

        # Prerequisites: Added a valid segment so it doesn't trigger the fallback!
        mocks["transcription_api"].get_transcript.return_value = Transcript(
            source_id=source_id,
            segments=[TranscriptSegment(start_sec=10.0, end_sec=20.0, text="Look at this graph.")]
        )

        # Pass 2: NLP Router
        mocks["nlp_router_api"].generate_vision_windows.return_value = [TimeWindow(start_sec=10.0, end_sec=20.0)]

        mocks["repo"].save_map.side_effect = lambda x: x

        result = api.process_multimodal_timeline(source_id, "vid.mp4")

        assert isinstance(result, IntelligentSceneMap)
        assert len(result.scenes) == 2
        assert result.scenes[0].extraction_method == "global_pass"
        assert result.scenes[1].extraction_method == "semantic_pass"
        assert mocks["scene_api"].process_video.call_count == 2