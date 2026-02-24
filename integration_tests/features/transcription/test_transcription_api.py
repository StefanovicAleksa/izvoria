# filename: test_transcription_api.py
# path: tests/integration/features/transcription/test_transcription_api.py
import os
import uuid
import pytest
import numpy as np
import soundfile as sf

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.storage.data.models import SourceModel, FileModel
from app.features.storage.domain.entities import SourceType

from app.features.vad.data.repositories import VadRepository
from app.features.vad.domain.entities import VadMap, VadSegment

from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
from app.features.transcription.data.repositories import TranscriptRepository
from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.transcription.domain.entities import Transcript


class TestTranscriptionAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL, NVIDIA GPU, and the Faster-Whisper model downloaded locally.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        """Ensures the database schema exists before testing."""
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield
        # We do not drop tables here to allow inspecting the DB if a test fails.

    @pytest.fixture
    def real_audio_file(self, tmp_path):
        """Generates a real 3-second 16kHz WAV file containing a simple sine wave."""
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # FIX: Explicitly cast the Numpy array to float32 before generating the WAV file
        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        file_path = tmp_path / "integration_test_audio.wav"

        # FIX: Explicitly instruct soundfile to write PCM_16 (standard FFmpeg output format)
        sf.write(file_path, audio_data, sample_rate, subtype='PCM_16')

        return str(file_path)

    @pytest.fixture
    def test_source_id(self):
        """Creates a real Source and File in the database to satisfy Foreign Key constraints."""
        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        with db_manager.session() as session:
            # 1. Create the prerequisite File record to satisfy the Foreign Key
            test_file = FileModel(
                id=file_id,
                file_path="/tmp/fake_integration_audio.wav",
                file_name="fake_integration_audio.wav",
                file_size_bytes=1024,
                mime_type="audio/wav"
            )
            session.add(test_file)

            # 2. Create the Source record
            test_source = SourceModel(
                id=source_id,
                name="integration_test_video.mp4",
                source_type=SourceType.VIDEO.value,
                content_hash=f"test-hash-{source_id}",
                file_id=file_id
            )
            session.add(test_source)

            # The context manager automatically commits both records safely

        return source_id

    @pytest.fixture
    def live_api(self):
        """Wires together the actual repositories, VRAM orchestrator, and AI adapter."""
        vad_repo = VadRepository(db_manager)
        transcript_repo = TranscriptRepository(db_manager)

        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        adapter = FasterWhisperAdapter()

        return TranscriptionAPI(
            repository=transcript_repo,
            vad_repository=vad_repo,
            orchestrator=orchestrator,
            adapter=adapter
        )

    def test_process_audio_end_to_end(self, live_api, test_source_id, real_audio_file):
        """
        Executes the Transcription phase end-to-end on real hardware.
        """
        # 1. Pre-requisite: Save a real VadMap to the database
        vad_repo = live_api._vad_repo
        test_vad_map = VadMap(
            source_id=test_source_id,
            segments=[
                # Tell the transcriber that the whole 3 seconds is "speech"
                VadSegment(start_sec=0.0, end_sec=3.0, confidence=0.99)
            ]
        )
        vad_repo.save_map(test_vad_map)

        # 2. Skip test if model isn't downloaded (prevents CI pipelines without models from crashing)
        if not os.path.exists(settings.WHISPER_MODEL_PATH):
            pytest.skip(f"Faster-Whisper model not found at {settings.WHISPER_MODEL_PATH}. Skipping hardware test.")

        # 3. Execute the API (This hits the GPU and the DB)
        transcript = live_api.process_audio(
            source_id=test_source_id,
            audio_path=real_audio_file,
            release_after=True
        )

        # 4. Verify Domain Object
        assert isinstance(transcript, Transcript)
        assert transcript.source_id == test_source_id

        # Whisper might transcribe a pure sine wave as empty text, "[Music]", or a hallucination.
        # The key is that it successfully processed the audio slice without crashing.
        assert isinstance(transcript.segments, list)

        # 5. Verify Database Persistence
        # Check that it actually made it into Postgres
        transcript_repo = live_api._repo
        db_transcript = transcript_repo.get_by_source_id(test_source_id)

        assert db_transcript is not None
        assert db_transcript.id == transcript.id
        assert len(db_transcript.segments) == len(transcript.segments)

        # 6. Verify Idempotency (Should not hit the GPU a second time)
        transcript_second_run = live_api.process_audio(
            source_id=test_source_id,
            audio_path=real_audio_file,
            release_after=True
        )
        assert transcript_second_run.id == transcript.id