# Canvas: test_batch_media_pipeline_integration_v10
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

# --- NEW DOMAIN ADAPTATION IMPORTS ---
from app.features.domain_adaptation.data.repositories import DomainAdaptationRepository
from app.features.domain_adaptation.data.scout_whisper_adapter import ScoutWhisperAdapter
from app.features.domain_adaptation.data.deberta_adapter import DebertaClassifierAdapter
from app.features.domain_adaptation.utils.audio_sampler import AudioSampler
from app.features.domain_adaptation.service.adaptation_api import DomainAdaptationAPI
from app.orchestration.service.phases.domain_adaptation_phase import DomainAdaptationPhase
# -------------------------------------

from app.features.transcription.data.repositories import TranscriptRepository
from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter

from app.features.diarization.data.repositories import DiarizationRepository
from app.features.diarization.service.diarization_api import DiarizationAPI
from app.features.diarization.data.nemo_adapter import NeMoDiarizationAdapter

from app.features.sentence_reconstruction.data.repositories import PostgresSentenceRepository
from app.features.sentence_reconstruction.service.reconstruction_api import SentenceReconstructionAPI
from app.features.sentence_reconstruction.data.wtp_adapter import SaTAdapter

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.phases.vad_phase import VadPhase
from app.orchestration.service.phases.transcribe_phase import TranscribePhase
from app.orchestration.service.phases.diarize_phase import DiarizePhase
from app.orchestration.service.phases.sentence_reconstruction_phase import SentenceReconstructionPhase
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

        # Real Domain Adaptation Feature
        domain_repo = DomainAdaptationRepository(db_manager)
        real_scout_adapter = ScoutWhisperAdapter()
        real_classifier_adapter = DebertaClassifierAdapter()
        sampler = AudioSampler(slice_duration_sec=20.0)
        domain_adaptation_api = DomainAdaptationAPI(
            repository=domain_repo,
            orchestrator=orchestrator,
            scout_adapter=real_scout_adapter,
            classifier_adapter=real_classifier_adapter,
            audio_sampler=sampler
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

        # Real Sentence Reconstruction Feature
        sentence_repo = PostgresSentenceRepository(db_manager)
        real_sat_adapter = SaTAdapter()
        reconstruction_api = SentenceReconstructionAPI(sentence_repo, orchestrator, real_sat_adapter)
        sat_manifest = ModelManifest(
            model_type=ModelType.SENTENCE_RECONSTRUCTION,
            model_path=settings.SAT_MODEL_NAME,
            required_vram_mb=settings.SAT_VRAM_REQUIREMENT_MB
        )

        # Phases
        scan_phase = ScanPhase(job_manager, scanner)
        ingest_phase = IngestPhase(job_manager, storage_api)
        extract_audio_phase = ExtractAudioPhase(job_manager, audio_extractor)
        vad_phase = VadPhase(job_manager, vad_api, storage_api, async_service, vad_manifest, real_vad_adapter)
        domain_adaptation_phase = DomainAdaptationPhase(job_manager, domain_adaptation_api, storage_api, vad_repo) # <-- NEW
        transcribe_phase = TranscribePhase(job_manager, transcription_api, storage_api, domain_repo, async_service, transcribe_manifest, real_transcribe_adapter) # <-- UPDATED
        diarize_phase = DiarizePhase(job_manager, diarization_api, storage_api, async_service, diarize_manifest, real_diarize_adapter)
        sentence_phase = SentenceReconstructionPhase(job_manager, reconstruction_api, transcription_api, async_service, sat_manifest, real_sat_adapter)

        pipeline = BatchMediaPipeline(
            scan_phase=scan_phase,
            ingest_phase=ingest_phase,
            extract_audio_phase=extract_audio_phase,
            vad_phase=vad_phase,
            domain_adaptation_phase=domain_adaptation_phase,
            transcribe_phase=transcribe_phase,
            diarize_phase=diarize_phase,
            sentence_reconstruction_phase=sentence_phase
        )

        yield pipeline, storage_api

        async_service.shutdown()

    def test_end_to_end_media_pipeline(self, pipeline_setup, tmp_path):
        """Tests the seamless flow from Directory Scan -> Domain Adapt -> DB Generating Transcript -> Reconstructed Sentences without mocks."""
        pipeline, storage_api = pipeline_setup

        # Dynamic Guard Clause verifying the SaT model exists before attempting hardware load
        model_path = settings.SAT_MODEL_NAME
        # Often loaded via cache or local folder depending on implementation, skip if not found
        if not os.path.exists(model_path) and not os.getenv("ALLOW_DOWNLOADS"):
             pytest.skip(f"SaT Model not found locally at {model_path} and downloads disabled. Skipping hardware integration test.")

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
        assert PipelinePhase.DOMAIN_ADAPTATION in results  # <-- Assert new phase ran
        assert PipelinePhase.TRANSCRIBE in results
        assert PipelinePhase.DIARIZE in results
        assert PipelinePhase.SENTENCE_RECONSTRUCTION in results

        assert results[PipelinePhase.SCAN].status == PhaseStatus.COMPLETED, f"Scan failed: {results[PipelinePhase.SCAN].errors}"
        assert results[PipelinePhase.INGEST].status == PhaseStatus.COMPLETED, f"Ingest failed: {results[PipelinePhase.INGEST].errors}"
        assert results[PipelinePhase.AUDIO_EXTRACTION].status == PhaseStatus.COMPLETED, f"Audio Extraction failed: {results[PipelinePhase.AUDIO_EXTRACTION].errors}"
        assert results[PipelinePhase.VAD].status == PhaseStatus.COMPLETED, f"VAD failed: {results[PipelinePhase.VAD].errors}"
        assert results[PipelinePhase.DOMAIN_ADAPTATION].status == PhaseStatus.COMPLETED, f"Domain Adaptation failed: {results[PipelinePhase.DOMAIN_ADAPTATION].errors}" # <-- Assert success
        assert results[PipelinePhase.TRANSCRIBE].status == PhaseStatus.COMPLETED, f"Transcribe failed: {results[PipelinePhase.TRANSCRIBE].errors}"
        assert results[PipelinePhase.DIARIZE].status == PhaseStatus.COMPLETED, f"Diarize failed: {results[PipelinePhase.DIARIZE].errors}"

        sentence_result = results[PipelinePhase.SENTENCE_RECONSTRUCTION]
        assert sentence_result.status == PhaseStatus.COMPLETED, f"Sentence Reconstruction failed: {sentence_result.errors}"
        assert sentence_result.successful_count == 1
        assert sentence_result.items[0].reconstructed_transcript_id is not None