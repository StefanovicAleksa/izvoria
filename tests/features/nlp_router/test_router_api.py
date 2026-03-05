# Canvas: test_router_api_v2
# Path: tests/features/nlp_router/test_router_api.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.nlp_router.domain.interfaces import ILLMRouterAdapter
from app.features.nlp_router.domain.entities import RouterReasoning, TimeWindow
from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.features.nlp_router.exceptions.base import NLPRouterError
from app.features.nlp_router.service.router_api import NLPRouterAPI


class TestNLPRouterAPI:

    @pytest.fixture
    def setup_api(self):
        mock_orchestrator = MagicMock(spec=ModelManagerOrchestrator)
        mock_adapter = MagicMock(spec=ILLMRouterAdapter)
        manifest = ModelManifest(model_type=ModelType.LLM, model_path="fake", required_vram_mb=100)

        # Setup acquire to return a dummy loaded model
        fake_loaded = LoadedModel(manifest=manifest, model_instance="fake_weights")
        mock_orchestrator.acquire_model.return_value = fake_loaded

        api = NLPRouterAPI(orchestrator=mock_orchestrator, adapter=mock_adapter, manifest=manifest)
        return api, mock_orchestrator, mock_adapter, fake_loaded

    def test_consolidate_windows_logic(self, setup_api):
        """Tests the mathematical gap threshold consolidation."""
        api, _, _, _ = setup_api

        raw_windows = [
            TimeWindow(start_sec=10.0, end_sec=20.0),
            TimeWindow(start_sec=22.0, end_sec=30.0),  # Within 5s gap, should merge
            TimeWindow(start_sec=40.0, end_sec=50.0)  # Far away, shouldn't merge
        ]

        consolidated = api._consolidate_windows(raw_windows, gap_threshold_sec=5.0)

        assert len(consolidated) == 2
        assert consolidated[0].start_sec == 10.0
        assert consolidated[0].end_sec == 30.0
        assert consolidated[1].start_sec == 40.0

    def test_generate_vision_windows_graceful_degradation(self, setup_api):
        """
        Tests that if the LLM permanently hallucinates on Chunk 1, the API skips it,
        continues to process Chunk 2, and releases VRAM safely.
        """
        api, mock_orchestrator, mock_adapter, fake_loaded = setup_api

        # Create a transcript long enough to force multiple chunks
        transcript = Transcript(
            source_id=uuid.uuid4(),
            segments=[
                TranscriptSegment(start_sec=0.0, end_sec=100.0, text="Chunk 1 Part 1"),
                TranscriptSegment(start_sec=100.0, end_sec=180.0, text="Chunk 1 Part 2"),
                TranscriptSegment(start_sec=180.0, end_sec=250.0, text="Chunk 2"),
            ]
        )

        # Chunk 1: Fail completely. Chunk 2: Succeed.
        success_result = RouterReasoning(
            reasoning="Success",
            windows=[TimeWindow(start_sec=190.0, end_sec=200.0)]
        )

        mock_adapter.extract_visual_cues.side_effect = [
            NLPRouterError("Hallucination!"),
            success_result
        ]

        # Execute
        windows = api.generate_vision_windows(transcript)

        # Assertions
        assert len(windows) == 1
        assert windows[0].start_sec == 190.0
        assert windows[0].end_sec == 205.0  # Includes the 5.0s camera buffer

        # Ensure VRAM was strictly released despite the internal error
        mock_orchestrator.release_model.assert_called_once_with(fake_loaded, mock_adapter, flush=True)