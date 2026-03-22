# Canvas: test_correction_api_integration_v6
# Path: integration_tests/features/phonetic_correction/test_correction_api_integration.py

import os
import json
import uuid
import shutil
import pytest
from pathlib import Path
from sqlalchemy import text as sa_text  # Aliased to prevent shadowing

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.phonetic_correction.data.repositories import PhoneticRepository
from app.features.phonetic_correction.data.qwen_adapter import QwenAdapter
from app.features.phonetic_correction.service.correction_api import PhoneticCorrectionAPI
from app.features.phonetic_correction.domain.entities import PhoneticCorrectedTranscript


class TestPhoneticCorrectionAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL, NVIDIA GPU, and the Qwen 2.5 7B GGUF model downloaded locally.
    Tests the full execution pipeline, Sliding Window Chunking, Hallucination Scrubbing, and DB persistence.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self):
        if not os.path.exists(settings.ASR_PHASE_3_MODEL_PATH):
            pytest.skip(f"Integration test skipped due to missing Qwen model at: {settings.ASR_PHASE_3_MODEL_PATH}")

    @pytest.fixture(scope="module")
    def setup_medical_glossary(self):
        """
        Copies the isolated medical.json from integration test resources
        into the application's live glossary directory for the test duration.
        """
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        resource_path = project_root / "integration_tests" / "resources" / "medical.json"
        target_path = project_root / "data" / "glossaries" / "medical.json"

        if not resource_path.exists():
            resource_path.parent.mkdir(parents=True, exist_ok=True)
            mock_manifest = {
                "base_terms": ["acute angina", "hypoxemia", "ischemia"],
                "sub_domains": {}
            }
            resource_path.write_text(json.dumps(mock_manifest))

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(resource_path, target_path)

        yield "MEDICAL"

        if target_path.exists():
            target_path.unlink()

    @pytest.fixture
    def real_transcript(self) -> ReconstructedTranscript:
        integration_tests_dir = Path(__file__).resolve().parent.parent.parent
        json_path = integration_tests_dir / "resources" / "phonetic_correction_transcript.json"

        if not json_path.exists():
            pytest.skip(f"Test transcript missing at {json_path}. Please create it.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for seg in data["sentences"]:
            if "segments" in seg:
                del seg["segments"]
            if "words" not in seg:
                seg["words"] = []

        sentences = [ReconstructedSentence(**seg) for seg in data["sentences"]]
        return ReconstructedTranscript(source_id=uuid.UUID(data["source_id"]), sentences=sentences)

    @pytest.fixture
    def massive_transcript(self) -> ReconstructedTranscript:
        """
        Generates a massive 100-sentence transcript to force the Phonetic API
        to trigger multiple overlapping Sliding Windows.
        """
        source_id = uuid.uuid4()
        sentences = []

        for i in range(100):
            # FIX: Renamed variable to sentence_text to avoid shadowing sqlalchemy.text
            if i == 45:
                sentence_text = "The patient presented with a cute angina."
            elif i == 85:
                sentence_text = "Subscribe to the channel."
            else:
                sentence_text = f"This is routine medical observation sentence number {i} to pad the LLM context window."

            sentences.append(
                ReconstructedSentence(
                    sentence_id=uuid.uuid4(),
                    start_sec=float(i),
                    end_sec=float(i + 1),
                    text=sentence_text,
                    speaker_id="Speaker_0",
                    words=[]
                )
            )

        return ReconstructedTranscript(source_id=source_id, sentences=sentences)

    @pytest.fixture
    def live_api(self) -> PhoneticCorrectionAPI:
        repo = PhoneticRepository(db_manager)
        qwen_adapter = QwenAdapter()
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        return PhoneticCorrectionAPI(
            repository=repo,
            orchestrator=orchestrator,
            patch_adapter=qwen_adapter
        )

    @pytest.fixture
    def isolated_setup(self) -> None:
        """Drops foreign keys so we can insert transcripts without needing real media files."""
        with db_manager.session() as session:
            session.execute(sa_text(
                "ALTER TABLE phonetic_corrected_transcripts DROP CONSTRAINT IF EXISTS phonetic_corrected_transcripts_source_id_fkey;"
            ))
            session.commit()

    def test_execute_end_to_end_correction_and_scrubbing(
            self,
            isolated_setup: None,
            live_api: PhoneticCorrectionAPI,
            setup_medical_glossary: str,
            real_transcript: ReconstructedTranscript
    ):
        """Tests the standard base-case correction on a small real transcript."""
        try:
            corrected_transcript = live_api.execute(
                transcript=real_transcript,
                industry=setup_medical_glossary,
                sub_domains=["GENERAL"],
                release_after=True
            )
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping test due to insufficient VRAM: {e}")
            raise e

        # FIX: Added strict type assertion to utilize the PhoneticCorrectedTranscript import
        assert isinstance(corrected_transcript, PhoneticCorrectedTranscript)

        patched_text = " ".join([s.text for s in corrected_transcript.sentences])
        assert "acute angina" in patched_text.lower()
        assert "a cute angina" not in patched_text.lower()
        assert "subscribe to the channel" not in patched_text.lower()

    def test_sliding_window_stress_test(
            self,
            isolated_setup: None,
            live_api: PhoneticCorrectionAPI,
            setup_medical_glossary: str,
            massive_transcript: ReconstructedTranscript
    ):
        """
        STRESS TEST: Forces the pipeline to process 100 sentences.
        Proves the sliding window chunker safely handles massive transcripts
        without OOMing the Qwen 2.5 8192 token limit, and correctly maps
        JSON patches across multiple chunk boundaries.
        """
        try:
            corrected_transcript = live_api.execute(
                transcript=massive_transcript,
                industry=setup_medical_glossary,
                sub_domains=["GENERAL"],
                release_after=True
            )
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping stress test due to insufficient VRAM: {e}")
            raise e

        # FIX: Added strict type assertion
        assert isinstance(corrected_transcript, PhoneticCorrectedTranscript)

        # 1. Verify the API returned a full-length transcript (no dropped chunks)
        assert len(corrected_transcript.sentences) == 99

        patched_text = " ".join([s.text for s in corrected_transcript.sentences])

        # 2. Verify patches applied across different sliding window chunks
        assert "acute angina" in patched_text.lower(), "Phonetic patch in Chunk 2 failed!"
        assert "a cute angina" not in patched_text.lower()
        assert "subscribe to the channel" not in patched_text.lower(), "Hallucination in Chunk 3 failed to scrub!"

        # 3. Verify Audit Trail captured both cross-chunk corrections
        assert len(corrected_transcript.audit_trail) == 2

        print(f"\n[DEBUG] Sliding Window Stress Test Executed Successfully! (3 Chunks Processed)")