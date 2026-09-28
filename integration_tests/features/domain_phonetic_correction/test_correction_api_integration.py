# Canvas: test_domain_correction_api_integration_v33
# Path: integration_tests/features/domain_phonetic_correction/test_correction_api_integration.py

import os
import json
import uuid
import shutil
import pytest
from pathlib import Path
from typing import Generator
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.domain_phonetic_correction.data.repositories import DomainPhoneticRepository
from app.features.domain_phonetic_correction.data.qwen_adapter import QwenDomainAdapter
from app.features.domain_phonetic_correction.service.correction_api import DomainPhoneticCorrectionAPI
from app.features.domain_phonetic_correction.domain.entities import DomainPhoneticCorrectedTranscript


class TestDomainPhoneticCorrectionAPIIntegration:
    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self) -> Generator[None, None, None]:
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self) -> None:
        if not os.path.exists(settings.DOMAIN_PHONETIC_CORRECTION_MODEL_PATH):
            pytest.skip(
                f"Integration test skipped due to missing Qwen model at: {settings.DOMAIN_PHONETIC_CORRECTION_MODEL_PATH}")

    @pytest.fixture(scope="module")
    def setup_medical_glossary(self) -> Generator[str, None, None]:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        resource_path = project_root / "integration_tests" / "resources" / "medical.json"
        target_path = settings.GLOSSARY_DIR / "medical.json"

        if not resource_path.exists():
            resource_path.parent.mkdir(parents=True, exist_ok=True)
            mock_manifest = {
                "base_terms": ["acute angina", "hypoxemia", "ischemia"],
                "sub_domains": {}
            }
            resource_path.write_text(json.dumps(mock_manifest))

        target_path.parent.mkdir(parents=True, exist_ok=True)
        original = target_path.read_bytes() if target_path.exists() else None
        shutil.copy(resource_path, target_path)
        yield "MEDICAL"
        # Restore the application's tracked glossary instead of deleting it
        if original is not None:
            target_path.write_bytes(original)
        elif target_path.exists():
            target_path.unlink()

    @pytest.fixture(scope="module")
    def setup_trading_glossary(self) -> Generator[str, None, None]:
        source_path = settings.BASE_DIR / "data" / "glossaries" / "en" / "prop_trading_macro.json"
        if not source_path.exists():
            pytest.skip(f"Trading glossary missing at {source_path}")
        target_path = settings.GLOSSARY_DIR / "prop_trading_macro.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        original = target_path.read_bytes() if target_path.exists() else None
        shutil.copy(source_path, target_path)
        yield "PROP_TRADING_MACRO"
        if original is not None:
            target_path.write_bytes(original)
        elif target_path.exists():
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
    def massive_array_transcript(self) -> ReconstructedTranscript:
        source_id = uuid.uuid4()
        sentences = []

        organs = ["heart", "liver", "kidney", "lung", "spleen", "pancreas", "stomach", "brain", "colon", "thyroid"]

        for i in range(100):
            if i == 20:
                # We inject '0' artifacts here to verify the Pipeline Sanitizer catches them
                sentence_text = "The 0patient 0presented with a cute angina0."
            else:
                sentence_text = f"The {organs[i % 10]} function test {i} returned completely normal metabolic parameters."

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
    def brutal_oversized_transcript(self) -> ReconstructedTranscript:
        source_id = uuid.uuid4()
        organs = ["heart", "liver", "kidney", "lung", "spleen", "pancreas", "stomach", "brain", "colon", "thyroid"]

        text_parts = ["The patient presented with a cute angina."]

        for i in range(150):
            text_parts.append(
                f"The {organs[i % 10]} function test {i} returned completely normal metabolic parameters.")

        massive_text = " ".join(text_parts)

        sentence = ReconstructedSentence(
            sentence_id=uuid.uuid4(),
            start_sec=0.0,
            end_sec=10.0,
            text=massive_text,
            speaker_id="Speaker_0",
            words=[]
        )

        return ReconstructedTranscript(source_id=source_id, sentences=[sentence])

    @pytest.fixture
    def live_api(self) -> DomainPhoneticCorrectionAPI:
        repo = DomainPhoneticRepository(db_manager)
        qwen_adapter = QwenDomainAdapter()
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        return DomainPhoneticCorrectionAPI(
            repository=repo,
            orchestrator=orchestrator,
            patch_adapter=qwen_adapter
        )

    @pytest.fixture
    def isolated_setup(self) -> None:
        with db_manager.session() as session:
            session.execute(sa_text(
                "ALTER TABLE domain_phonetic_corrected_transcripts DROP CONSTRAINT IF EXISTS domain_phonetic_corrected_transcripts_source_id_fkey;"
            ))
            session.commit()

    def test_execute_end_to_end_domain_correction(
            self,
            isolated_setup: None,
            live_api: DomainPhoneticCorrectionAPI,
            setup_medical_glossary: str,
            real_transcript: ReconstructedTranscript
    ) -> None:
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

        assert isinstance(corrected_transcript, DomainPhoneticCorrectedTranscript)

        patched_text = " ".join([s.text for s in corrected_transcript.sentences])
        assert "acute angina" in patched_text.lower()
        assert "a cute angina" not in patched_text.lower()

    def test_sliding_window_array_stress_test_with_sanitization(
            self,
            isolated_setup: None,
            live_api: DomainPhoneticCorrectionAPI,
            setup_medical_glossary: str,
            massive_array_transcript: ReconstructedTranscript
    ) -> None:
        try:
            corrected_transcript = live_api.execute(
                transcript=massive_array_transcript,
                industry=setup_medical_glossary,
                sub_domains=["GENERAL"],
                release_after=True
            )
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping stress test due to insufficient VRAM: {e}")
            raise e

        assert isinstance(corrected_transcript, DomainPhoneticCorrectedTranscript)

        patched_text = " ".join([s.text for s in corrected_transcript.sentences])

        # Assert that the text was patched AND the 0s were fully sanitized out
        assert "acute angina" in patched_text.lower(), "Phonetic patch in later chunk failed!"
        assert "a cute angina" not in patched_text.lower()
        assert "0patient" not in patched_text.lower()
        assert "angina0" not in patched_text.lower()
        assert len(corrected_transcript.audit_trail) >= 1

    def test_trading_speech_is_left_alone_and_a_real_mishearing_is_fixed(
            self,
            isolated_setup: None,
            live_api: DomainPhoneticCorrectionAPI,
            setup_trading_glossary: str
    ) -> None:
        # Natural speech from the 74-minute Axia video that the old free-rewrite pass damaged, plus one mis-hearing
        untouched = [
            "So again showing that there is still a little bit of weakness in the bund.",
            "Looking at a chart, your first thought is: well, you have the fade from the top of the range up at those 73s.",
            "You want to set your replay for 9th March, starting at 13.50.45 seconds.",
            "Past three days and including the morning of the Bank of England decision, the market has rallied.",
        ]
        misheard = "So, as I mentioned before, this is where Mario Draggy can use the questions to inform the market."
        sentences = [
            ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=float(i), end_sec=i + 1.0, text=text,
                                  speaker_id="speaker_0", words=[])
            for i, text in enumerate(untouched + [misheard])
        ]
        transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=sentences)

        try:
            corrected = live_api.execute(
                transcript=transcript,
                industry=setup_trading_glossary,
                sub_domains=["CENTRAL_BANK_MACRO", "EUROPEAN_CENTRAL_BANK"],
                release_after=True
            )
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping test due to insufficient VRAM: {e}")
            raise e

        assert [s.text for s in corrected.sentences] == untouched + [misheard.replace("Mario Draggy", "Mario Draghi")]
        assert [(p.original, p.correction) for r in corrected.audit_trail for p in r.applied_patches] == [
            ("Mario Draggy", "Mario Draghi")
        ]

    def test_oversized_sentence_is_corrected_in_place(
            self,
            isolated_setup: None,
            live_api: DomainPhoneticCorrectionAPI,
            setup_medical_glossary: str,
            brutal_oversized_transcript: ReconstructedTranscript
    ) -> None:
        try:
            corrected_transcript = live_api.execute(
                transcript=brutal_oversized_transcript,
                industry=setup_medical_glossary,
                sub_domains=["GENERAL"],
                release_after=True
            )
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping stress test due to insufficient VRAM: {e}")
            raise e

        assert isinstance(corrected_transcript, DomainPhoneticCorrectedTranscript)

        # Sentences are no longer split for the model: the verifier sees a window around each candidate
        assert len(corrected_transcript.sentences) == 1

        patched_text = " ".join([s.text for s in corrected_transcript.sentences])

        assert "acute angina" in patched_text.lower(), "Phonetic patch failed on the oversized sentence!"
        assert "a cute angina" not in patched_text.lower()
        assert len(corrected_transcript.audit_trail) >= 1