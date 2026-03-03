# Canvas: test_bridge_api_integration_v4
# Path: integration_tests/features/intelligence_bridge/test_bridge_api_integration.py

import os
import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

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

from app.features.scene_detector.data.ffmpeg_scene_adapter import FFmpegSceneAdapter
from app.features.scene_detector.utils.phash_deduplicator import PhashDeduplicator
from app.features.scene_detector.service.scene_api import SceneDetectorAPI

from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.transcription.domain.entities import Transcript, TranscriptSegment

from app.features.nlp_router.data.phi3_adapter import Phi3RouterAdapter
from app.features.nlp_router.service.router_api import NLPRouterAPI

from app.features.intelligence_bridge.data.repository import IntelligentSceneRepository
from app.features.intelligence_bridge.service.bridge_api import IntelligenceBridgeAPI
from app.features.intelligence_bridge.domain.entities import IntelligentSceneMap


class TestIntelligenceBridgeAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST (Except for Transcription).
    Requires: Live PostgreSQL, FFmpeg, NVIDIA GPU, and Phi-3.5-mini GGUF.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = global_db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(autouse=True)
    def check_ffmpeg_installed(self):
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except FileNotFoundError:
            pytest.skip("FFmpeg is not installed on this system.")

    @pytest.fixture
    def api_setup(self, db_manager, tmp_path):
        """Wires up all REAL dependencies for the Intelligence Bridge (Transcription is mocked)."""
        # 1. Storage (Required for DB Foreign Keys)
        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        # 2. VRAM & Orchestrator
        monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(monitor)

        # 3. Scene Detector (Real FFmpeg extraction, completely DB-free now)
        ffmpeg_adapter = FFmpegSceneAdapter(timeout_seconds=60)
        deduplicator = PhashDeduplicator(hamming_distance_threshold=5)
        scene_api = SceneDetectorAPI(
            ffmpeg_adapter=ffmpeg_adapter,
            storage_api=storage_api,
            deduplicator=deduplicator
        )

        # 4. Transcription (Mocked as permitted to avoid unneeded VAD/Whisper loading errors)
        transcription_api = MagicMock(spec=TranscriptionAPI)

        # 5. NLP Router (Real Phi-3.5 hardware inference)
        phi_adapter = Phi3RouterAdapter()
        model_path = getattr(settings, "PHI_MODEL_PATH", "models/Phi-3.5-mini-instruct-Q4_K_M.gguf")
        phi_manifest = ModelManifest(
            model_type=ModelType.LLM,
            model_path=model_path,
            required_vram_mb=3500.0
        )
        nlp_router_api = NLPRouterAPI(orchestrator, phi_adapter, phi_manifest)

        # 6. Intelligence Bridge
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
        """4-sec video: 2s black, 2s white to guarantee a visual cut."""
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

        if not os.path.exists(bridge_api._nlp_router_api._manifest.model_path):
            pytest.skip(f"Phi-3.5 model not found. Skipping hardware integration test.")

        # 1. Create and Ingest Video (Satisfies Foreign Key constraints)
        source_dir = tmp_path / "test_media"
        source_dir.mkdir()
        video_file = source_dir / "semantic_cut.mp4"
        self.generate_hard_cut_video(video_file)

        video_source = storage_api.ingest_local_file(
            file_path=os.fspath(video_file),
            source_type=SourceType.VIDEO
        )
        source_id = video_source.id

        # 2. Inject a mocked transcript (Bypassing the DB and Whisper logic)
        # The physical cut is at 2.0s. Let's make the speaker say "Look at this graph" at 1.5s - 2.5s.
        mock_transcription_api.get_transcript.return_value = Transcript(
            source_id=source_id,
            segments=[
                TranscriptSegment(start_sec=0.0, end_sec=1.5, text="Hello world."),
                TranscriptSegment(start_sec=1.5, end_sec=2.5, text="Look at this graph on the screen."),
                TranscriptSegment(start_sec=2.5, end_sec=4.0, text="It is very interesting.")
            ]
        )

        # 3. Execute the full Multimodal Pipeline
        result_map = bridge_api.process_multimodal_timeline(
            source_id=source_id,
            video_path=video_source.file.file_path
        )

        # 4. Assertions
        assert isinstance(result_map, IntelligentSceneMap)
        assert result_map.source_id == source_id

        # We expect Phi-3.5 to flag the phrase, triggering a semantic extraction
        # around the 2.0s mark. Deduplication should keep the semantic tag.
        assert len(result_map.scenes) > 0

        # Verify the DB vaulted the correct output structure
        db_fetch = bridge_api._repo.get_map_by_source_id(source_id)
        assert db_fetch is not None
        assert len(db_fetch.scenes) == len(result_map.scenes)

        semantic_scenes = [s for s in result_map.scenes if s.extraction_method == "semantic_pass"]
        assert len(semantic_scenes) >= 1, "Expected the real LLM to extract a semantic scene."