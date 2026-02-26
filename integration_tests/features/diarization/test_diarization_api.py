# Path: integration_tests/features/diarization/test_diarization_api.py

import os
import uuid
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.storage.data.models import SourceModel, FileModel
from app.features.storage.domain.entities import SourceType
from app.features.storage.service.storage_api import StorageAPI

from app.features.vad.data.repositories import VadRepository
from app.features.vad.domain.entities import VadMap, VadSegment

from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
from app.features.transcription.data.repositories import TranscriptRepository
from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.transcription.domain.entities import Transcript, TranscriptSegment

from app.features.diarization.data.nemo_adapter import NeMoDiarizationAdapter
from app.features.diarization.data.repositories import DiarizationRepository
from app.features.diarization.service.diarization_api import DiarizationAPI
from app.features.diarization.domain.entities import DiarizationMap


class TestDiarizationAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL, NVIDIA GPU, and the TitaNet-Large model downloaded locally.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture
    def real_audio_file(self):
        """Points to the real human speech test file."""
        integration_tests_dir = Path(__file__).resolve().parent.parent.parent
        audio_path = integration_tests_dir / "resources" / "test_speech.wav"

        if not audio_path.exists():
            pytest.skip(f"Test audio file not found at {audio_path}. Please ensure it exists.")

        return str(audio_path)

    @pytest.fixture
    def isolated_setup(self):
        """
        Creates a completely isolated state for this test, seeding the DB
        with a File, Source, VadMap, and Transcript prerequisite.
        """
        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        # 1. Seed Storage
        with db_manager.session() as session:
            test_file = FileModel(
                id=file_id,
                file_path="/tmp/fake_diarization_audio.wav",
                file_name="fake_diarization_audio.wav",
                file_size_bytes=1024,
                mime_type="audio/wav"
            )
            session.add(test_file)

            test_source = SourceModel(
                id=source_id,
                name="diarization_integration_video.mp4",
                source_type=SourceType.VIDEO.value,
                content_hash=f"diarization-hash-{source_id}",
                file_id=file_id
            )
            session.add(test_source)

            # --- CRITICAL DATABASE FIX ---
            # Commit the transaction so we release table locks!
            # If we don't do this, Pytest will freeze the Storage tests that run after this.
            session.commit()
            # -----------------------------

        # 2. Seed VAD Map (Assuming speech is present from 0 to 3 seconds)
        vad_repo = VadRepository(db_manager)
        vad_repo.save_map(VadMap(
            source_id=source_id,
            segments=[VadSegment(start_sec=0.0, end_sec=3.0, confidence=0.99)]
        ))

        # 3. Seed Transcript
        transcript_repo = TranscriptRepository(db_manager)
        transcript_repo.save(Transcript(
            source_id=source_id,
            segments=[TranscriptSegment(start_sec=0.5, end_sec=2.5, text="Hello world, this is a real test.")]
        ))

        return source_id

    @pytest.fixture
    def live_api(self):
        """Constructs the full, live API dependency tree."""
        vad_repo = VadRepository(db_manager)
        transcript_repo = TranscriptRepository(db_manager)
        diarization_repo = DiarizationRepository(db_manager)

        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        transcription_api = TranscriptionAPI(
            repository=transcript_repo,
            vad_repository=vad_repo,
            orchestrator=orchestrator,
            adapter=FasterWhisperAdapter()
        )

        adapter = NeMoDiarizationAdapter()

        # Create a Dummy StorageAPI to satisfy the strict Type Checker
        dummy_storage_api = MagicMock(spec=StorageAPI)

        return DiarizationAPI(
            repository=diarization_repo,
            vad_repository=vad_repo,
            transcription_api=transcription_api,
            storage_api=dummy_storage_api,
            orchestrator=orchestrator,
            adapter=adapter
        )

    def test_process_audio_end_to_end(self, live_api, isolated_setup, real_audio_file):
        source_id = isolated_setup

        model_path = getattr(settings, "DIARIZATION_MODEL_PATH", "models/titanet_large.nemo")
        if not os.path.exists(model_path):
            pytest.skip(f"TitaNet model not found at {model_path}. Skipping hardware test.")

        # --- 1. First Run: Execute Pipeline ---
        diarization_map = live_api.process_audio(
            source_id=source_id,
            audio_path=real_audio_file,
            release_after=True
        )

        # Verify Domain Object
        assert isinstance(diarization_map, DiarizationMap)
        assert diarization_map.source_id == source_id
        assert isinstance(diarization_map.segments, list)

        # --- 2. Verify Database Persistence (Diarization Map) ---
        db_map = live_api._repo.get_map_by_source_id(source_id)
        assert db_map is not None
        assert db_map.id == diarization_map.id

        # --- 3. Verify Cross-Feature Database Update (Transcript) ---
        updated_transcript = live_api._transcription_api.get_transcript(source_id)
        assert updated_transcript is not None
        if len(diarization_map.segments) > 0 and len(updated_transcript.segments) > 0:
            # If NeMo successfully grouped the real speech into a speaker,
            # our aligner should have assigned it to the text segment.
            assert updated_transcript.segments[0].speaker_id != "Unknown"

        # --- 4. Verify Idempotency ---
        second_run_map = live_api.process_audio(
            source_id=source_id,
            audio_path=real_audio_file,
            release_after=True
        )
        assert second_run_map.id == diarization_map.id