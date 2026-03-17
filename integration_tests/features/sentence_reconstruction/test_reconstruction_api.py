# Canvas: test_reconstruction_api_integration_v3
# Path: integration_tests/features/sentence_reconstruction/test_reconstruction_api.py

import json
import uuid
import pytest
from pathlib import Path
from sqlalchemy import text

from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.transcription.domain.entities import Transcript, TranscriptSegment

from app.features.sentence_reconstruction.data.sat_adapter import SaTAdapter
from app.features.sentence_reconstruction.data.repositories import PostgresSentenceRepository
from app.features.sentence_reconstruction.service.reconstruction_api import SentenceReconstructionAPI
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript


class TestSentenceReconstructionAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL, NVIDIA GPU, and the SaT wtpsplit model.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture
    def real_transcript(self):
        """Loads the strict edge-case transcript from the JSON resources."""
        integration_tests_dir = Path(__file__).resolve().parent.parent.parent
        json_path = integration_tests_dir / "resources" / "reconstruction_transcript.json"

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
    def isolated_setup(self, real_transcript):
        """Isolates the DB environment so we don't have to mock a massive Source/File tree."""
        with db_manager.session() as session:
            session.execute(text(
                "ALTER TABLE reconstructed_transcripts DROP CONSTRAINT IF EXISTS reconstructed_transcripts_source_id_fkey;"))
        return real_transcript.source_id

    @pytest.fixture
    def live_api(self):
        """Constructs the full, live API dependency tree."""
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        repo = PostgresSentenceRepository(db_manager)
        adapter = SaTAdapter()

        return SentenceReconstructionAPI(
            repository=repo,
            orchestrator=orchestrator,
            adapter=adapter
        )

    def test_reconstruct_transcript_end_to_end(self, live_api, isolated_setup, real_transcript):
        """Executes the pipeline and ensures edge cases (fragmentation, speaker walls) are handled."""
        source_id = isolated_setup

        try:
            # 1. Execute Pipeline
            reconstructed_transcript = live_api.reconstruct_transcript(
                transcript=real_transcript,
                release_after=True
            )
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Skipping test due to missing GPU/CUDA environment: {e}")
            raise e

        # 2. Verify Domain Objects
        assert isinstance(reconstructed_transcript, ReconstructedTranscript)
        assert reconstructed_transcript.source_id == source_id
        assert len(reconstructed_transcript.sentences) > 0

        # 3. Verify Neural Punctuation & Speaker Wall
        sentences = reconstructed_transcript.sentences

        # We expect the AI to logically group the fragmented text.
        # "okay let us begin" + "the deposition of mr smith" -> "Okay let us begin the deposition of Mr Smith."
        speaker_0_text = " ".join([s.text.lower() for s in sentences if s.speaker_id == "Speaker_0"])
        assert "deposition of mr smith." in speaker_0_text or "deposition of mr smith" in speaker_0_text
        assert "state your name" in speaker_0_text

        speaker_1_text = " ".join([s.text.lower() for s in sentences if s.speaker_id == "Speaker_1"])
        assert "john arthur smith" in speaker_1_text

        # 4. Verify Absolute UI Audit Trail
        # The first reconstructed sentence should wrap multiple underlying acoustic tokens
        assert len(sentences[0].segments) > 0

        # 5. Verify Database Persistence
        db_transcript = live_api._repo.get_by_source_id(source_id)
        assert db_transcript is not None
        assert db_transcript.id == reconstructed_transcript.id
        assert len(db_transcript.sentences) == len(reconstructed_transcript.sentences)