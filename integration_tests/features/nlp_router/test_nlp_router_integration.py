# Canvas: test_nlp_router_integration_v1
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
    Requires: NVIDIA GPU, CUDA Toolkit, and Phi-3.5-mini GGUF downloaded.
    """

    @pytest.fixture
    def real_transcript(self):
        """Loads the engineered test transcript from the resources folder."""
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
        """Wires up the REAL VRAM Monitor, Orchestrator, and llama.cpp Adapter."""
        monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor=monitor)
        adapter = Phi3RouterAdapter()

        # Safely fetch the model path (fallback to the manual download location)
        model_path = getattr(settings, "PHI_MODEL_PATH", "models/Phi-3.5-mini-instruct-Q4_K_M.gguf")

        manifest = ModelManifest(
            model_type=ModelType.TRANSCRIPTION,  # Reusing existing Enum for testing
            model_path=model_path,
            required_vram_mb=3500.0  # 3.5GB VRAM footprint
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
        # Skip if the user hasn't downloaded the 2.2GB GGUF file
        if not os.path.exists(live_api._manifest.model_path):
            pytest.skip(f"Phi-3.5 model not found at {live_api._manifest.model_path}. Skipping hardware test.")

        print("\n[INFO] Booting Phi-3.5 into VRAM... This will take a few seconds.")

        # Execute the real LLM inference
        windows = live_api.generate_vision_windows(real_transcript)

        # Basic Assertions
        assert isinstance(windows, list)
        for w in windows:
            assert isinstance(w, TimeWindow)

        # We expect at least one consolidated window since the transcript has obvious visual cues.
        assert len(windows) > 0, "The LLM failed to extract any visual windows from the transcript."

        # --- ADVANCED INTELLIGENCE ASSERTIONS ---

        # 1. Did it ignore the metaphor?
        # At 0.0s - 10.0s, the speaker says "look at the big picture".
        # The LLM should NOT extract this. The first window should start around 10.0s.
        first_window = windows[0]
        assert first_window.start_sec >= 9.0, (
            f"Failed: The LLM hallucinated and extracted the metaphorical 'big picture' phrase. "
            f"It started extraction at {first_window.start_sec}s"
        )

        # 2. Did it catch the literal charts?
        # At 15.0s, it says "revenue chart". At 35.0s, it says "This graph".
        # Due to our 5-second gap threshold consolidation, these might be merged into one massive
        # window (10.0s to 45.0s), or split into two.
        # Either way, the final window must cover the 40.0s mark (plus the 5s camera buffer).
        last_window = windows[-1]
        assert last_window.end_sec >= 40.0, (
            f"Failed: The LLM missed the 'graph' reference at the end of the transcript. "
            f"It stopped extraction at {last_window.end_sec}s"
        )

        print(f"\n[SUCCESS] Extracted Visual Windows: {windows}")