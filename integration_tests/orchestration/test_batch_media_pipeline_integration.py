# filename: test_batch_media_pipeline_integration_v2.py
# path: integration_tests/orchestration/test_batch_media_pipeline_integration.py
import os
import subprocess
import pytest
import uuid
from pathlib import Path
from unittest.mock import Mock

from app.core.jobs.service.manager import JobManager
from app.core.jobs.data.repositories import JobRepository

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel, VramStatus
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.service.async_model_service import AsyncModelService

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType

from app.features.scanner.service.scanner import FileScanner

from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI
from app.features.audio_extractor.data.repositories import AudioExtractionRepository
from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter

from app.features.vad.data.repositories import VadRepository
from app.features.vad.service.vad_api import VadAPI
from app.features.vad.domain.interfaces import IVadModelAdapter
from app.features.vad.domain.entities import VadMap

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.phases.vad_phase import VadPhase
from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus


class DummyVadAdapter(IVadModelAdapter):
    """Bypasses NeMo and CUDA to simulate VAD processing for integration tests."""

    def load(self, manifest: ModelManifest) -> LoadedModel:
        return LoadedModel(manifest=manifest, model_instance="fake_nemo_model")

    def unload(self, model: LoadedModel) -> bool:
        return True

    def generate_vad_map(self, model: LoadedModel, audio_path: str, source_id: uuid.UUID) -> VadMap:
        return VadMap(source_id=source_id, segments=[])


class TestBatchMediaPipelineIntegration:
    @pytest.fixture(autouse=True)
    def check_ffmpeg_installed(self):
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except FileNotFoundError:
            pytest.skip("FFmpeg is not installed on this system.")

    @pytest.fixture
    def pipeline_setup(self, db_manager, tmp_path):
        """Wires up ALL real dependencies with the real integration test database, simulating hardware layers."""
        job_repo = JobRepository(db_manager)
        job_manager = JobManager(job_repo)

        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        scanner = FileScanner()

        audio_repo = AudioExtractionRepository(db_manager)
        ffmpeg_adapter = FFmpegAdapter(timeout_seconds=30)
        audio_extractor = AudioExtractorAPI(
            ffmpeg_adapter=ffmpeg_adapter,
            repository=audio_repo,
            storage_api=storage_api
        )

        # Mock VRAM hardware to bypass CUDA
        mock_vram_monitor = Mock()
        mock_vram_monitor.get_vram_status.return_value = VramStatus(total_vram_mb=12000, used_vram_mb=0,
                                                                    free_vram_mb=12000)

        # Real Model Orchestration Stack
        orchestrator = ModelManagerOrchestrator(mock_vram_monitor)
        bg_worker = BackgroundModelWorker(orchestrator)
        async_service = AsyncModelService(bg_worker)

        # Real VAD Feature (with Dummy AI model)
        vad_repo = VadRepository(db_manager)
        dummy_vad_adapter = DummyVadAdapter()
        vad_api = VadAPI(vad_repo, orchestrator, dummy_vad_adapter)

        vad_manifest = ModelManifest(model_type=ModelType.VAD, model_path="fake", required_vram_mb=100)

        # Phases
        scan_phase = ScanPhase(job_manager, scanner)
        ingest_phase = IngestPhase(job_manager, storage_api)
        extract_audio_phase = ExtractAudioPhase(job_manager, audio_extractor)
        vad_phase = VadPhase(
            job_manager=job_manager,
            vad_api=vad_api,
            storage_api=storage_api,
            async_model_service=async_service,
            vad_manifest=vad_manifest,
            vad_adapter=dummy_vad_adapter
        )

        pipeline = BatchMediaPipeline(
            scan_phase=scan_phase,
            ingest_phase=ingest_phase,
            extract_audio_phase=extract_audio_phase,
            vad_phase=vad_phase
        )

        yield pipeline, storage_api

        async_service.shutdown()

    @staticmethod
    def generate_real_dummy_video(output_path: Path) -> None:
        command: list[str] = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=10x10:d=1",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1",
            os.fspath(output_path)
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_end_to_end_media_pipeline(self, pipeline_setup, tmp_path):
        """Tests the seamless flow from Directory Scan -> VAD Map DB Generation."""
        pipeline, storage_api = pipeline_setup

        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()
        valid_video = source_dir / "real_test_video.mp4"
        self.generate_real_dummy_video(valid_video)

        # ---------------------------------------------------------
        # EXECUTE THE PIPELINE
        # ---------------------------------------------------------
        results = pipeline.process_directory(str(source_dir), SourceType.VIDEO)

        # ---------------------------------------------------------
        # VERIFY STATE MACHINE RESULTS
        # ---------------------------------------------------------
        assert PipelinePhase.SCAN in results
        assert PipelinePhase.INGEST in results
        assert PipelinePhase.AUDIO_EXTRACTION in results
        assert PipelinePhase.VAD in results

        assert results[PipelinePhase.SCAN].status == PhaseStatus.COMPLETED
        assert results[PipelinePhase.INGEST].status == PhaseStatus.COMPLETED
        assert results[PipelinePhase.AUDIO_EXTRACTION].status == PhaseStatus.COMPLETED

        vad_result = results[PipelinePhase.VAD]
        assert vad_result.status == PhaseStatus.COMPLETED
        assert vad_result.successful_count == 1
        assert vad_result.items[0].vad_map_id is not None