# Canvas: test_correction_api_integration_v1
# Path: integration_tests/features/hallucination_correction/test_correction_api_integration.py

import json
import uuid
import pytest
from pathlib import Path

from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.features.storage.data.models import SourceModel, FileModel
from app.features.storage.domain.entities import SourceType

from app.features.vad.data.repositories import VadRepository
from app.features.vad.domain.entities import VadMap, VadSegment

from app.features.transcription.data.repositories import TranscriptRepository
from app.features.transcription.domain.entities import Transcript, TranscriptSegment

from app.features.hallucination_correction.service.correction_api import CorrectionAPI


class TestHallucinationCorrectionAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL connection.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture
    def fixture_data(self) -> dict:
        """Loads the hallucinated transcript JSON fixture."""
        integration_tests_dir = Path(__file__).resolve().parent.parent.parent
        fixture_path = integration_tests_dir / "resources" / "test_hallucinated_transcript.json"

        if not fixture_path.exists():
            pytest.skip(f"Test fixture not found at {fixture_path}.")

        with open(fixture_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def isolated_setup(self, fixture_data) -> uuid.UUID:
        """Seeds the PostgreSQL database with the Source, VAD Map, and dirty Transcript."""
        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        # 1. Seed Storage (Required for Foreign Keys)
        with db_manager.session() as session:
            test_file = FileModel(
                id=file_id,
                file_path="/tmp/fake_correction_audio.wav",
                file_name="fake_correction_audio.wav",
                file_size_bytes=1024,
                mime_type="audio/wav"
            )
            session.add(test_file)

            test_source = SourceModel(
                id=source_id,
                name="correction_integration_video.mp4",
                source_type=SourceType.VIDEO.value,
                content_hash=f"correction-hash-{source_id}",
                file_id=file_id
            )
            session.add(test_source)

        # 2. Seed VAD Map
        vad_repo = VadRepository(db_manager)
        vad_segments = [VadSegment(**seg) for seg in fixture_data["vad_segments"]]
        vad_repo.save_map(VadMap(source_id=source_id, segments=vad_segments))

        # 3. Seed Dirty Transcript
        transcript_repo = TranscriptRepository(db_manager)
        transcript_segments = [TranscriptSegment(**seg) for seg in fixture_data["transcript_segments"]]
        transcript_repo.save(Transcript(source_id=source_id, segments=transcript_segments))

        return source_id

    @pytest.fixture
    def live_api(self) -> CorrectionAPI:
        """Constructs the API with live database repositories."""
        vad_repo = VadRepository(db_manager)
        transcript_repo = TranscriptRepository(db_manager)

        return CorrectionAPI(
            transcript_repository=transcript_repo,
            vad_repository=vad_repo
        )

    def test_process_transcript_end_to_end(self, live_api, isolated_setup):
        """
        Executes the API boundary and verifies it correctly scrubs the
        dirty records from the Postgres database.
        """
        source_id = isolated_setup

        # --- 1. Execute the Pipeline ---
        cleaned_transcript, telemetry = live_api.process_transcript(source_id)

        # --- 2. Verify Telemetry ---
        assert telemetry.total_original_segments == 4
        assert telemetry.removed_by_vad == 1
        assert telemetry.removed_by_metadata == 1
        assert telemetry.removed_by_regex == 1
        assert telemetry.total_surviving_segments == 1

        # --- 3. Verify Returned Domain Object ---
        assert len(cleaned_transcript.segments) == 1
        assert cleaned_transcript.segments[0].text == "This is a completely valid and normal sentence."

        # --- 4. Verify Database Persistence ---
        # Fetch a fresh copy from the database to ensure the JSONB column was actually overwritten
        db_transcript = live_api._transcript_repo.get_by_source_id(source_id)
        assert db_transcript is not None
        assert len(db_transcript.segments) == 1
        assert db_transcript.segments[0].text == "This is a completely valid and normal sentence."

    def test_process_transcript_idempotency(self, live_api, isolated_setup):
        """Ensures running the scrubber twice on the same record doesn't delete valid data."""
        source_id = isolated_setup

        # Run 1
        live_api.process_transcript(source_id)

        # Run 2
        cleaned_transcript, telemetry = live_api.process_transcript(source_id)

        # Telemetry should now show 1 original segment and 0 removals
        assert telemetry.total_original_segments == 1
        assert telemetry.removed_by_vad == 0
        assert telemetry.removed_by_metadata == 0
        assert telemetry.removed_by_regex == 0
        assert len(cleaned_transcript.segments) == 1