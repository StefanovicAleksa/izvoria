# Canvas: test_nlp_router_integration_v5
# Path: integration_tests/features/nlp_router/test_nlp_router_integration.py

import os
import json
import uuid
import pytest
import torch
from pathlib import Path
from transformers import set_seed

from app.core.config import settings
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.legacy_features.scene_detector.domain.entities import TimeWindow
from app.legacy_features.nlp_router.data.phi3_adapter import Phi3RouterAdapter
from app.legacy_features.nlp_router.service.router_api import NLPRouterAPI


class TestNLPRouterIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: NVIDIA GPU, CUDA Toolkit, and local Hugging Face LLM.
    """

    @pytest.fixture(autouse=True)
    def enforce_determinism(self):
        """Forces the Hugging Face generation to be 100% deterministic for tests."""
        set_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

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
        model_path = live_api._manifest.model_path
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            pytest.skip(f"No HF model found at {model_path}. Skipping hardware test.")

        print("\n[INFO] Booting LLM into VRAM... This will take a few seconds.")

        windows = live_api.generate_vision_windows(real_transcript)

        assert isinstance(windows, list)
        for w in windows:
            assert isinstance(w, TimeWindow)

        assert len(windows) > 0, "The LLM failed to extract any visual windows and degraded gracefully."

        first_window = windows[0]
        assert first_window.start_sec >= 9.0

        last_window = windows[-1]
        assert last_window.end_sec >= 40.0

        print(f"\n[SUCCESS] Extracted Visual Windows: {windows}")