# Canvas: test_bridge_api_integration_v9
# Path: integration_tests/features/intelligence_bridge/test_bridge_api_integration.py

import os
import pytest
import subprocess
import torch
from pathlib import Path
from unittest.mock import MagicMock
from transformers import set_seed

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager as global_db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType

from app.legacy_features.scene_detector.data.ffmpeg_scene_adapter import FFmpegSceneAdapter
from app.legacy_features.scene_detector.utils.phash_deduplicator import PhashDeduplicator
from app.legacy_features.scene_detector.service.scene_api import SceneDetectorAPI

from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.transcription.domain.entities import Transcript, TranscriptSegment

from app.legacy_features.nlp_router.data.phi3_adapter import Phi3RouterAdapter
from app.legacy_features.nlp_router.service.router_api import NLPRouterAPI

from app.legacy_features.intelligence_bridge.data.repository import IntelligentSceneRepository
from app.legacy_features.intelligence_bridge.service.bridge_api import IntelligenceBridgeAPI
from app.legacy_features.intelligence_bridge.domain.entities import IntelligentSceneMap


class TestIntelligenceBridgeAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST
    (Except for Transcription).
    Requires: Live PostgreSQL, FFmpeg, NVIDIA GPU, and local Hugging Face LLM.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = global_db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(autouse=True)
    def enforce_determinism_and_ffmpeg(self):
        set_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except FileNotFoundError:
            pytest.skip("FFmpeg is not installed on this system.")

    @pytest.fixture
    def api_setup(self, db_manager, tmp_path):
        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(monitor)

        ffmpeg_adapter = FFmpegSceneAdapter(timeout_seconds=60)
        deduplicator = PhashDeduplicator(hamming_distance_threshold=5)
        scene_api = SceneDetectorAPI(
            ffmpeg_adapter=ffmpeg_adapter,
            storage_api=storage_api,
            deduplicator=deduplicator
        )

        transcription_api = MagicMock(spec=TranscriptionAPI)
        phi_adapter = Phi3RouterAdapter()

        phi_manifest = ModelManifest(
            model_type=ModelType.LLM,
            model_path=settings.NLP_MODEL_PATH,
            required_vram_mb=settings.NLP_VRAM_REQUIREMENT_MB
        )
        nlp_router_api = NLPRouterAPI(orchestrator, phi_adapter, phi_manifest)

        bridge_repo = IntelligentSceneRepository(db_manager)
        bridge_api = IntelligenceBridgeAPI(
            repository=bridge_repo,
            scene_api=scene_api,
            transcription_api=transcription_api,
            nlp_router_api=nlp_router_api
        )

        return bridge_api, storage_api, transcription_api

    @staticmethod
    def generate_hard_cut_video(output_path: Path) -> None:
        command: list[str] = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2",
            "-f", "lavfi", "-i", "color=c=white:s=320x240:d=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out_video]",
            "-map", "[out_video]",
            os.fspath(output_path)
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_end_to_end_multimodal_timeline(self, api_setup, tmp_path):
        bridge_api, storage_api, mock_transcription_api = api_setup

        model_path = bridge_api._nlp_router_api._manifest.model_path
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            pytest.skip(f"No HF model found at {model_path}. Skipping hardware integration test.")

        source_dir = tmp_path / "test_media"
        source_dir.mkdir()
        video_file = source_dir / "semantic_cut.mp4"
        self.generate_hard_cut_video(video_file)

        video_source = storage_api.ingest_local_file(
            file_path=os.fspath(video_file),
            source_type=SourceType.VIDEO
        )
        source_id = video_source.id

        mock_transcription_api.get_transcript.return_value = Transcript(
            source_id=source_id,
            segments=[
                TranscriptSegment(start_sec=0.0, end_sec=1.5, text="Hello world."),
                TranscriptSegment(start_sec=1.5, end_sec=2.5, text="Look at this graph on the screen."),
                TranscriptSegment(start_sec=2.5, end_sec=4.0, text="It is very interesting.")
            ]
        )

        result_map = bridge_api.process_multimodal_timeline(
            source_id=source_id,
            video_path=video_source.file.file_path
        )

        assert isinstance(result_map, IntelligentSceneMap)
        assert result_map.source_id == source_id
        assert len(result_map.scenes) > 0

        db_fetch = bridge_api._repo.get_map_by_source_id(source_id)
        assert db_fetch is not None
        assert len(db_fetch.scenes) == len(result_map.scenes)

        semantic_scenes = [s for s in result_map.scenes if s.extraction_method == "semantic_pass"]
        assert len(semantic_scenes) >= 1