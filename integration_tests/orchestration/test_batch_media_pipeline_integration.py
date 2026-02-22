# filename: test_batch_media_pipeline_integration_v1.py
import os
import subprocess
import pytest
from pathlib import Path

from app.core.jobs.service.manager import JobManager
from app.core.jobs.data.repositories import JobRepository

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType

from app.features.scanner.service.scanner import FileScanner

from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI
from app.features.audio_extractor.data.repositories import AudioExtractionRepository
from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus


class TestBatchMediaPipelineIntegration:
    @pytest.fixture(autouse=True)
    def check_ffmpeg_installed(self):
        """Skip these integration tests if FFmpeg is not installed on the host machine."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        except FileNotFoundError:
            pytest.skip("FFmpeg is not installed on this system. Skipping pipeline integration test.")

    @pytest.fixture
    def pipeline_setup(self, db_manager, tmp_path):
        """
        Wires up ALL real dependencies with the real integration test database.
        """
        # 1. Core Jobs
        job_repo = JobRepository(db_manager)
        job_manager = JobManager(job_repo)

        # 2. Storage Feature (using tmp_path for the vault)
        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        # 3. Scanner Feature
        scanner = FileScanner()

        # 4. Audio Extractor Feature
        audio_repo = AudioExtractionRepository(db_manager)
        ffmpeg_adapter = FFmpegAdapter(timeout_seconds=30)
        audio_extractor = AudioExtractorAPI(
            ffmpeg_adapter=ffmpeg_adapter,
            repository=audio_repo,
            storage_api=storage_api
        )

        # 5. Orchestration Phases
        scan_phase = ScanPhase(job_manager, scanner)
        ingest_phase = IngestPhase(job_manager, storage_api)
        extract_audio_phase = ExtractAudioPhase(job_manager, audio_extractor)

        # 6. Pipeline Director
        pipeline = BatchMediaPipeline(
            scan_phase=scan_phase,
            ingest_phase=ingest_phase,
            extract_audio_phase=extract_audio_phase
        )

        return pipeline, storage_api

    @staticmethod
    def generate_real_dummy_video(output_path: Path) -> None:
        """Uses FFmpeg to generate a real 1-second blank video with silent audio."""
        command: list[str] = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=10x10:d=1",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1",
            os.fspath(output_path)
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_end_to_end_media_pipeline(self, pipeline_setup, tmp_path):
        """
        Scenario:
        1. A real folder contains 1 valid video and 1 invalid text file.
        2. The pipeline scans, ingests the video, and automatically extracts the audio.
        3. The DB job history and physical storage vault are perfectly synced.
        """
        pipeline, storage_api = pipeline_setup

        # ---------------------------------------------------------
        # PHASE 0: PREPARE THE RAW DATA
        # ---------------------------------------------------------
        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()

        # File 1: Valid Video
        valid_video = source_dir / "real_test_video.mp4"
        self.generate_real_dummy_video(valid_video)

        # File 2: Invalid File (Extension not allowed by scanner)
        invalid_script = source_dir / "malicious.exe"
        invalid_script.write_bytes(b"bad bytes")

        # ---------------------------------------------------------
        # EXECUTE THE PIPELINE
        # ---------------------------------------------------------
        # We explicitly tell the pipeline that this batch is VIDEO
        results = pipeline.process_directory(str(source_dir), SourceType.VIDEO)

        # ---------------------------------------------------------
        # VERIFY STATE MACHINE RESULTS
        # ---------------------------------------------------------
        assert PipelinePhase.SCAN in results
        assert PipelinePhase.INGEST in results
        assert PipelinePhase.AUDIO_EXTRACTION in results

        scan_result = results[PipelinePhase.SCAN]
        ingest_result = results[PipelinePhase.INGEST]
        extract_result = results[PipelinePhase.AUDIO_EXTRACTION]

        # Scan should have found exactly 1 file (the mp4), ignoring the exe
        assert scan_result.status == PhaseStatus.COMPLETED
        assert scan_result.successful_count == 1

        # Ingest should have successfully ingested the 1 video
        assert ingest_result.status == PhaseStatus.COMPLETED
        assert ingest_result.successful_count == 1
        video_source_id = ingest_result.items[0].source_id
        assert video_source_id is not None

        # Extract should have successfully generated 1 audio file from the video
        assert extract_result.status == PhaseStatus.COMPLETED
        assert extract_result.successful_count == 1
        extracted_audio_id = extract_result.items[0].extracted_file_id
        assert extracted_audio_id is not None

        # ---------------------------------------------------------
        # VERIFY PHYSICAL VAULT
        # ---------------------------------------------------------
        vault_files = list(storage_api.file_manager.base_path.rglob("*.*"))

        # There should be exactly 2 physical files in the vault:
        # 1. The ingested video (.mp4)
        # 2. The extracted audio (.wav)
        assert len(vault_files) == 2

        extensions = [f.suffix for f in vault_files]
        assert ".mp4" in extensions
        assert ".wav" in extensions

        # ---------------------------------------------------------
        # VERIFY DATABASE LINEAGE
        # ---------------------------------------------------------
        # Ensure the Source record is intact
        saved_source = storage_api.get_source(video_source_id)
        assert saved_source is not None
        assert saved_source.source_type == SourceType.VIDEO
        assert saved_source.name == "real_test_video.mp4"