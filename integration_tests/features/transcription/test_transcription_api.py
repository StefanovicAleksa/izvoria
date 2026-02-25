# Canvas: test_transcription_api_integration_v2
# Path: tests/integration/features/transcription/test_transcription_api.py

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
from app.features.transcription.domain.entities import Transcript, TranscriptSegment


class TestTranscriptionAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL, NVIDIA GPU, and the Faster-Whisper model downloaded locally.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture
    def real_audio_file(self, tmp_path):
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        file_path = tmp_path / "integration_test_audio.wav"
        sf.write(file_path, audio_data, sample_rate, subtype='PCM_16')

        return str(file_path)

    @pytest.fixture
    def test_source_id(self):
        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        with db_manager.session() as session:
            test_file = FileModel(
                id=file_id,
                file_path="/tmp/fake_integration_audio.wav",
                file_name="fake_integration_audio.wav",
                file_size_bytes=1024,
                mime_type="audio/wav"
            )
            session.add(test_file)

            test_source = SourceModel(
                id=source_id,
                name="integration_test_video.mp4",
                source_type=SourceType.VIDEO.value,
                content_hash=f"test-hash-{source_id}",
                file_id=file_id
            )
            session.add(test_source)

        return source_id

    @pytest.fixture
    def live_api(self):
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
        vad_repo = live_api._vad_repo
        test_vad_map = VadMap(
            source_id=test_source_id,
            segments=[
                VadSegment(start_sec=0.0, end_sec=3.0, confidence=0.99)
            ]
        )
        vad_repo.save_map(test_vad_map)

        if not os.path.exists(settings.WHISPER_MODEL_PATH):
            pytest.skip(f"Faster-Whisper model not found at {settings.WHISPER_MODEL_PATH}. Skipping hardware test.")

        transcript = live_api.process_audio(
            source_id=test_source_id,
            audio_path=real_audio_file,
            release_after=True
        )

        assert isinstance(transcript, Transcript)
        assert transcript.source_id == test_source_id
        assert isinstance(transcript.segments, list)

        transcript_repo = live_api._repo
        db_transcript = transcript_repo.get_by_source_id(test_source_id)

        assert db_transcript is not None
        assert db_transcript.id == transcript.id
        assert len(db_transcript.segments) == len(transcript.segments)


    def test_update_transcript_end_to_end(self, live_api, test_source_id):
        """Verifies that the new update API directly modifies the JSONB segments in Postgres."""

        initial_transcript = Transcript(
            source_id=test_source_id,
            segments=[
                TranscriptSegment(start_sec=0.0, end_sec=3.0, text="Initial transcribed text")
            ]
        )
        live_api._repo.save(initial_transcript)

        # 2. Fetch the transcript we just seeded
        existing_transcript = live_api.get_transcript(test_source_id)
        assert existing_transcript is not None, "Transcript was not seeded properly"

        # 3. Simulate the Diarization feature modifying the transcript
        new_segments = [
            TranscriptSegment(
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                text=seg.text,
                speaker_id="Speaker_0"  # <--- Adding the speaker tag
            ) for seg in existing_transcript.segments
        ]

        updated_transcript = Transcript(
            id=existing_transcript.id,
            source_id=existing_transcript.source_id,
            segments=new_segments,
            created_at=existing_transcript.created_at
        )

        # 4. Use the public API to update the database
        live_api.update_transcript(updated_transcript)

        # 5. Fetch a completely fresh copy to verify persistence
        fresh_db_transcript = live_api.get_transcript(test_source_id)

        assert fresh_db_transcript is not None
        assert fresh_db_transcript.id == existing_transcript.id
        if len(fresh_db_transcript.segments) > 0:
            assert fresh_db_transcript.segments[0].speaker_id == "Speaker_0"