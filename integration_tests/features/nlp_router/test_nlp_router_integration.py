# Canvas: test_nlp_router_integration_v4
# Path: integration_tests/features/nlp_router/test_nlp_router_integration.py

import os
import json
import uuid
import pytest
from pathlib import Path

from app.core.config import settings
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.features.scene_detector.domain.entities import TimeWindow
from app.features.nlp_router.data.phi3_adapter import Phi3RouterAdapter
from app.features.nlp_router.service.router_api import NLPRouterAPI


class TestNLPRouterIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: NVIDIA GPU, CUDA Toolkit, and local Hugging Face LLM.
    """

    @pytest.fixture
    def real_transcript(self):
        integration_tests_dir = Path(__file__).resolve().parent.parent.parent
        json_path = integration_tests_dir / "resources" / "test_transcript.json"

        if not json_path.exists():
            pytest.skip(f"Test transcript missing at {json_path}. Please create it.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = [TranscriptSegment(**seg) for seg in data["segments"]]

        return Transcript(
            source_id=uuid.UUID(data["source_id"]),
            segments=segments
        )

    @pytest.fixture
    def live_api(self):
        monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor=monitor)
        adapter = Phi3RouterAdapter()

        manifest = ModelManifest(
            model_type=ModelType.LLM,
            model_path=settings.NLP_MODEL_PATH,
            required_vram_mb=settings.NLP_VRAM_REQUIREMENT_MB
        )

        return NLPRouterAPI(
            orchestrator=orchestrator,
            adapter=adapter,
            manifest=manifest
        )

    def test_end_to_end_vision_routing(self, live_api, real_transcript):
        """
        Scenario:
        1. Loads the 8-segment transcript.
        2. LLM evaluates metaphors vs. literal visual cues.
        3. Asserts the 5-second buffer and consolidation logic worked perfectly.
        """
        # FIX: Check for standard Hugging Face model structure (config.json)
        model_path = live_api._manifest.model_path
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            pytest.skip(f"No HF model found at {model_path}. Skipping hardware test.")

        print("\n[INFO] Booting LLM into VRAM... This will take a few seconds.")

        windows = live_api.generate_vision_windows(real_transcript)

        assert isinstance(windows, list)
        for w in windows:
            assert isinstance(w, TimeWindow)

        assert len(windows) > 0, "The LLM failed to extract any visual windows from the transcript."

        first_window = windows[0]
        assert first_window.start_sec >= 9.0, (
            f"Failed: The LLM hallucinated and extracted the metaphorical 'big picture' phrase.\n"
            f"It started extraction at {first_window.start_sec}s"
        )

        last_window = windows[-1]
        assert last_window.end_sec >= 40.0, (
            f"Failed: The LLM missed the 'graph' reference at the end of the transcript. "
            f"It stopped extraction at {last_window.end_sec}s"
        )

        print(f"\n[SUCCESS] Extracted Visual Windows: {windows}")