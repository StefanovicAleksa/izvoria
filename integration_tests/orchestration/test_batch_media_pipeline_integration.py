# Canvas: test_batch_media_pipeline_integration_v8
# Path: integration_tests/orchestration/test_batch_media_pipeline_integration.py

import os
import subprocess
import pytest
import shutil
from pathlib import Path

from app.core.config import settings
from app.core.jobs.service.manager import JobManager
from app.core.jobs.data.repositories import JobRepository

from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType

from app.features.scanner.service.scanner import FileScanner

from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI
from app.features.audio_extractor.data.repositories import AudioExtractionRepository
from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter

from app.features.vad.data.repositories import VadRepository
from app.features.vad.service.vad_api import VadAPI
from app.features.vad.data.nemo_adapter import NeMoVadAdapter

from app.features.transcription.data.repositories import TranscriptRepository
from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter

from app.features.diarization.data.repositories import DiarizationRepository
from app.features.diarization.service.diarization_api import DiarizationAPI
from app.features.diarization.data.nemo_adapter import NeMoDiarizationAdapter

from app.features.scene_detector.data.ffmpeg_scene_adapter import FFmpegSceneAdapter
from app.features.scene_detector.utils.phash_deduplicator import PhashDeduplicator
from app.features.scene_detector.service.scene_api import SceneDetectorAPI

from app.features.nlp_router.data.phi3_adapter import Phi3RouterAdapter
from app.features.nlp_router.service.router_api import NLPRouterAPI

from app.features.intelligence_bridge.data.repository import IntelligentSceneRepository
from app.features.intelligence_bridge.service.bridge_api import IntelligenceBridgeAPI

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.phases.vad_phase import VadPhase
from app.orchestration.service.phases.transcribe_phase import TranscribePhase
from app.orchestration.service.phases.diarize_phase import DiarizePhase
from app.orchestration.service.phases.intelligence_bridge_phase import IntelligenceBridgePhase
from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus


class TestBatchMediaPipelineIntegration:
    @pytest.fixture(autouse=True)
    def check_ffmpeg_installed(self):
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except FileNotFoundError:
            pytest.skip("FFmpeg is not installed on this system.")

    @pytest.fixture
    def pipeline_setup(self, db_manager, tmp_path):
        """Wires up ALL real dependencies with the real integration test database, requiring real GPU VRAM."""
        job_repo = JobRepository(db_manager)
        job_manager = JobManager(job_repo)

        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        scanner = FileScanner()

        audio_repo = AudioExtractionRepository(db_manager)
        ffmpeg_audio_adapter = FFmpegAdapter(timeout_seconds=300)
        audio_extractor = AudioExtractorAPI(
            ffmpeg_adapter=ffmpeg_audio_adapter,
            repository=audio_repo,
            storage_api=storage_api
        )

        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)
        bg_worker = BackgroundModelWorker(orchestrator)
        async_service = AsyncModelService(bg_worker)

        # Real VAD Feature
        vad_repo = VadRepository(db_manager)
        real_vad_adapter = NeMoVadAdapter()
        vad_api = VadAPI(vad_repo, orchestrator, real_vad_adapter)
        vad_manifest = ModelManifest(
            model_type=ModelType.VAD,
            model_path=settings.VAD_MODEL_PATH,
            required_vram_mb=settings.VAD_VRAM_REQUIREMENT_MB
        )

        # Real Transcription Feature
        transcript_repo = TranscriptRepository(db_manager)
        real_transcribe_adapter = FasterWhisperAdapter()
        transcription_api = TranscriptionAPI(transcript_repo, vad_repo, orchestrator, real_transcribe_adapter)
        transcribe_manifest = ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path=settings.WHISPER_MODEL_PATH,
            required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
        )

        # Real Diarization Feature
        diarize_repo = DiarizationRepository(db_manager)
        real_diarize_adapter = NeMoDiarizationAdapter()
        diarization_api = DiarizationAPI(diarize_repo, vad_repo, transcription_api, storage_api, orchestrator, real_diarize_adapter)
        diarize_manifest = ModelManifest(
            model_type=ModelType.DIARIZATION,
            model_path=settings.DIARIZATION_MODEL_PATH,
            required_vram_mb=settings.DIARIZATION_VRAM_REQUIREMENT_MB
        )

        # Real Intelligence Bridge Feature
        scene_adapter = FFmpegSceneAdapter(timeout_seconds=300)
        phash_dedup = PhashDeduplicator(hamming_distance_threshold=5)
        scene_api = SceneDetectorAPI(scene_adapter, storage_api, phash_dedup)

        # FIX: Ensure it dynamically inherits from settings to pass the missing HF models guard check
        nlp_manifest = ModelManifest(
            model_type=ModelType.LLM,
            model_path=settings.NLP_MODEL_PATH,
            required_vram_mb=settings.NLP_VRAM_REQUIREMENT_MB
        )
        real_nlp_adapter = Phi3RouterAdapter()
        nlp_router_api = NLPRouterAPI(orchestrator, real_nlp_adapter, nlp_manifest)

        bridge_repo = IntelligentSceneRepository(db_manager)
        bridge_api = IntelligenceBridgeAPI(bridge_repo, scene_api, transcription_api, nlp_router_api)

        # Phases
        scan_phase = ScanPhase(job_manager, scanner)
        ingest_phase = IngestPhase(job_manager, storage_api)
        extract_audio_phase = ExtractAudioPhase(job_manager, audio_extractor)
        vad_phase = VadPhase(job_manager, vad_api, storage_api, async_service, vad_manifest, real_vad_adapter)
        transcribe_phase = TranscribePhase(job_manager, transcription_api, storage_api, async_service, transcribe_manifest, real_transcribe_adapter)
        diarize_phase = DiarizePhase(job_manager, diarization_api, storage_api, async_service, diarize_manifest, real_diarize_adapter)
        bridge_phase = IntelligenceBridgePhase(job_manager, bridge_api, storage_api, async_service, nlp_manifest, real_nlp_adapter)

        pipeline = BatchMediaPipeline(
            scan_phase=scan_phase,
            ingest_phase=ingest_phase,
            extract_audio_phase=extract_audio_phase,
            vad_phase=vad_phase,
            transcribe_phase=transcribe_phase,
            diarize_phase=diarize_phase,
            intelligence_bridge_phase=bridge_phase
        )

        yield pipeline, storage_api

        async_service.shutdown()

    def test_end_to_end_media_pipeline(self, pipeline_setup, tmp_path):
        """Tests the seamless flow from Directory Scan -> DB Generating Transcript -> Intelligence Bridge Map without mocks."""
        pipeline, storage_api = pipeline_setup

        # FIX: Dynamic Guard Clause verifying the Hugging Face structure exists before attempting hardware load
        model_path = settings.NLP_MODEL_PATH
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            pytest.skip(f"No HF model found at {model_path}. Skipping hardware integration test.")

        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()

        original_vid = Path("integration_tests/resources/test_video.mp4").resolve()
        if not original_vid.exists():
            pytest.skip(f"Required test file not found at: {original_vid}")

        test_file = source_dir / "test_video.mp4"
        shutil.copy2(original_vid, test_file)

        # EXECUTE THE PIPELINE
        results = pipeline.process_directory(str(source_dir), SourceType.VIDEO)

        # VERIFY STATE MACHINE RESULTS
        assert PipelinePhase.SCAN in results
        assert PipelinePhase.INGEST in results
        assert PipelinePhase.AUDIO_EXTRACTION in results
        assert PipelinePhase.VAD in results
        assert PipelinePhase.TRANSCRIBE in results
        assert PipelinePhase.DIARIZE in results
        assert PipelinePhase.INTELLIGENCE_BRIDGE in results

        assert results[PipelinePhase.SCAN].status == PhaseStatus.COMPLETED, f"Scan failed: {results[PipelinePhase.SCAN].errors}"
        assert results[PipelinePhase.INGEST].status == PhaseStatus.COMPLETED, f"Ingest failed: {results[PipelinePhase.INGEST].errors}"
        assert results[PipelinePhase.AUDIO_EXTRACTION].status == PhaseStatus.COMPLETED, f"Audio Extraction failed: {results[PipelinePhase.AUDIO_EXTRACTION].errors}"
        assert results[PipelinePhase.VAD].status == PhaseStatus.COMPLETED, f"VAD failed: {results[PipelinePhase.VAD].errors}"
        assert results[PipelinePhase.TRANSCRIBE].status == PhaseStatus.COMPLETED, f"Transcribe failed: {results[PipelinePhase.TRANSCRIBE].errors}"
        assert results[PipelinePhase.DIARIZE].status == PhaseStatus.COMPLETED, f"Diarize failed: {results[PipelinePhase.DIARIZE].errors}"

        bridge_result = results[PipelinePhase.INTELLIGENCE_BRIDGE]
        assert bridge_result.status == PhaseStatus.COMPLETED, f"Intelligence Bridge failed: {bridge_result.errors}"
        assert bridge_result.successful_count == 1
        assert bridge_result.items[0].scene_map_id is not None