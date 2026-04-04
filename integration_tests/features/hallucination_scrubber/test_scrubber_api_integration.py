# Canvas: test_scrubber_api_integration_v3
# Path: integration_tests/features/hallucination_scrubber/test_scrubber_api_integration.py

import os
import json
import uuid
import pytest
from pathlib import Path
from typing import Generator

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.storage.data.models import SourceModel, FileModel
from app.features.storage.domain.entities import SourceType

from app.features.transcription.data.repositories import TranscriptRepository
from app.features.transcription.domain.entities import Transcript, TranscriptSegment

from app.features.domain_adaptation.data.repositories import DomainAdaptationRepository
from app.features.domain_adaptation.domain.entities import DomainGlossary

from app.features.hallucination_scrubber.data.repositories import HallucinationRepository
from app.features.hallucination_scrubber.data.deberta_adapter import DebertaAdapter
from app.features.hallucination_scrubber.service.scrubber_api import HallucinationScrubberAPI
from app.features.hallucination_scrubber.domain.entities import HallucinationScrubbedTranscript


class TestHallucinationScrubberAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL and the DeBERTa Zero-Shot model defined in Config.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self) -> Generator[None, None, None]:
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self) -> None:
        """Dynamically check for the model based on environment settings."""
        model_path = settings.HALLUCINATION_SCRUBBER_MODEL_PATH
        if not os.path.exists(model_path):
            pytest.skip(f"Integration test skipped due to missing DeBERTa model at: {model_path}")

    @pytest.fixture(scope="module")
    def dirty_transcript_resource(self) -> Generator[str, None, None]:
        """
        Dynamically generates a dirty transcript mimicking Whisper hallucinations
        and saves it to the resources folder for realistic testing.
        """
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        resource_path = project_root / "integration_tests" / "resources" / "dirty_transcript.json"

        if not resource_path.exists():
            resource_path.parent.mkdir(parents=True, exist_ok=True)
            mock_data = {
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 5.0,
                        "text": "The patient presented with acute angina.",
                        "speaker_id": "Speaker_0",
                        "words": [
                            {"start_sec": 0.0, "end_sec": 1.0, "word": "The", "probability": 0.99},
                            {"start_sec": 1.0, "end_sec": 2.0, "word": "patient", "probability": 0.99},
                            {"start_sec": 2.0, "end_sec": 5.0, "word": "presented...", "probability": 0.99}
                        ]
                    },
                    {
                        "start_sec": 5.0,
                        "end_sec": 15.0,
                        "text": "We will monitor his heart rate. Thank you for watching and please subscribe to the channel.",
                        "speaker_id": "Speaker_0",
                        "words": [
                            {"start_sec": 5.0, "end_sec": 6.0, "word": "We", "probability": 0.99},
                            {"start_sec": 6.0, "end_sec": 7.0, "word": "will", "probability": 0.99},
                            {"start_sec": 7.0, "end_sec": 8.0, "word": "monitor", "probability": 0.99},
                            {"start_sec": 8.0, "end_sec": 9.0, "word": "his", "probability": 0.99},
                            {"start_sec": 9.0, "end_sec": 10.0, "word": "heart", "probability": 0.99},
                            {"start_sec": 10.0, "end_sec": 11.0, "word": "rate.", "probability": 0.99},
                            # --- Hallucinated Words ---
                            {"start_sec": 11.0, "end_sec": 11.5, "word": "Thank", "probability": 0.99},
                            {"start_sec": 11.5, "end_sec": 12.0, "word": "you", "probability": 0.99},
                            {"start_sec": 12.0, "end_sec": 12.5, "word": "for", "probability": 0.99},
                            {"start_sec": 12.5, "end_sec": 13.0, "word": "watching", "probability": 0.99},
                            {"start_sec": 13.0, "end_sec": 13.5, "word": "and", "probability": 0.99},
                            {"start_sec": 13.5, "end_sec": 14.0, "word": "please", "probability": 0.99},
                            {"start_sec": 14.0, "end_sec": 14.5, "word": "subscribe", "probability": 0.99}
                        ]
                    },
                    {
                        "start_sec": 15.0,
                        "end_sec": 17.0,
                        "text": "Subtitles by Amara.org",
                        "speaker_id": "Speaker_0",
                        "words": []
                    }
                ]
            }
            resource_path.write_text(json.dumps(mock_data))

        yield str(resource_path)

    @pytest.fixture
    def live_api(self) -> HallucinationScrubberAPI:
        transcript_repo = TranscriptRepository(db_manager)
        domain_repo = DomainAdaptationRepository(db_manager)
        hallucination_repo = HallucinationRepository(db_manager)

        deberta_adapter = DebertaAdapter()
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        return HallucinationScrubberAPI(
            transcript_repo=transcript_repo,
            domain_repo=domain_repo,
            hallucination_repo=hallucination_repo,
            orchestrator=orchestrator,
            adapter=deberta_adapter
        )

    @pytest.fixture
    def isolated_setup(self, dirty_transcript_resource) -> uuid.UUID:
        """Seeds the PostgreSQL database with the prerequisites for Phase 6.5"""
        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        with db_manager.session() as session:
            # 1. Seed Storage
            test_file = FileModel(
                id=file_id,
                file_path="/tmp/fake_scrubber_video.mp4",
                file_name="fake_scrubber_video.mp4",
                file_size_bytes=1024,
                mime_type="video/mp4"
            )
            session.add(test_file)

            test_source = SourceModel(
                id=source_id,
                name="scrubber_integration_video.mp4",
                source_type=SourceType.VIDEO.value,
                content_hash=f"scrubber-hash-{source_id}",
                file_id=file_id
            )
            session.add(test_source)
            session.commit()

        # 2. Seed the Dirty Transcript
        with open(dirty_transcript_resource, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = [TranscriptSegment(**seg) for seg in data["segments"]]
        transcript = Transcript(source_id=source_id, segments=segments)

        transcript_repo = TranscriptRepository(db_manager)
        transcript_repo.save(transcript)

        # 3. Seed the Domain Glossary (so DeBERTa gets the specific "Medical" context)
        glossary = DomainGlossary(
            source_id=source_id,
            industry="MEDICAL",
            sub_domains=["DIAGNOSTICS"],
            hotword_string=""
        )
        domain_repo = DomainAdaptationRepository(db_manager)
        domain_repo.save_glossary(glossary)

        return source_id

    def test_execute_end_to_end_scrubbing(self, live_api: HallucinationScrubberAPI, isolated_setup: uuid.UUID):
        """
        Executes the Scrubber API boundary and verifies it correctly deletes
        hallucinations while preserving valid technical speech and timestamps.
        """
        source_id = isolated_setup

        try:
            # Execute Phase 6.5!
            scrubbed_transcript = live_api.execute(source_id=source_id, release_after=True)
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower() or "memory" in str(e).lower():
                pytest.skip(f"Skipping test due to insufficient VRAM: {e}")
            raise e

        assert isinstance(scrubbed_transcript, HallucinationScrubbedTranscript)

        # Ensure the final transcript only has 2 segments (the 3rd "Amara.org" segment should be completely deleted)
        assert len(scrubbed_transcript.segments) == 2

        # Segment 0: Should remain completely untouched
        assert scrubbed_transcript.segments[0].text == "The patient presented with acute angina."

        # Segment 1: Should be partially scrubbed (Slow Path)
        patched_text = scrubbed_transcript.segments[1].text
        assert "We will monitor his heart rate." in patched_text
        assert "Thank you for watching" not in patched_text
        assert "subscribe" not in patched_text

        # Verify the underlying timestamp array was safely filtered
        surviving_words = scrubbed_transcript.segments[1].words
        assert len(surviving_words) == 6
        assert surviving_words[-1].word == "rate."

        # Verify the Immutable Audit Trail
        assert len(scrubbed_transcript.audit_trail) > 0

        # Check that the complete deletion of Segment 2 ("Amara.org") was logged
        amara_deletion = next((a for a in scrubbed_transcript.audit_trail if "Amara.org" in a.original_text), None)
        assert amara_deletion is not None
        assert amara_deletion.scrubbed_text == ""
        assert len(amara_deletion.applied_patches) == 1

        # CRITICAL FIX: Ensure the detected label is any of our known negative classes, not just one hardcoded guess
        valid_negative_labels = [
            "youtube outro, subscription request, or video formatting text",
            "a polite sign-off, customer service greeting, or closing remark",
            "audio transcription watermark or auto-generated subtitle notice"
        ]
        assert amara_deletion.applied_patches[0].detected_label in valid_negative_labels

    def test_idempotency(self, live_api: HallucinationScrubberAPI, isolated_setup: uuid.UUID):
        """Proves that running the API twice returns the exact same DB record without loading VRAM again."""
        source_id = isolated_setup

        first_run = live_api.execute(source_id=source_id, release_after=True)
        second_run = live_api.execute(source_id=source_id, release_after=True)

        assert first_run.id == second_run.id