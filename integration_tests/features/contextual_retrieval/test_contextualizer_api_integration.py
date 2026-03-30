# Canvas: test_contextualizer_api_integration_v3
# Path: integration_tests/features/contextual_retrieval/test_contextualizer_api_integration.py

import os
import uuid
import pytest
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.phonetic_correction.domain.entities import PhoneticCorrectedTranscript

from app.features.contextual_retrieval.data.repositories import ContextualRepository
from app.features.contextual_retrieval.data.slm_adapter import SLMContextAdapter
from app.features.contextual_retrieval.service.contextualizer_api import ContextualizerAPI
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk


class TestContextualizerAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL, NVIDIA GPU, and the Llama 3.2 1B/3B Instruct GGUF model downloaded locally.
    Tests the full Contextual Retrieval Pipeline:
      1. Deterministic Sentence Chunking
      2. KV Caching and sliding window SLM Context Generation
      3. JSONB Database Persistence
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self):
        # We are reusing the DOMAIN_CLASSIFIER model path (Llama 3.2) as defined in our architecture
        if not os.path.exists(settings.DOMAIN_CLASSIFIER_MODEL_PATH):
            pytest.skip(f"Integration test skipped due to missing SLM at: {settings.DOMAIN_CLASSIFIER_MODEL_PATH}")

    @pytest.fixture
    def isolated_setup(self) -> uuid.UUID:
        """Isolates the DB environment so we don't have to mock a massive Source/File tree."""
        source_id = uuid.uuid4()
        with db_manager.session() as session:
            session.execute(sa_text(
                "ALTER TABLE contextualized_transcripts DROP CONSTRAINT IF EXISTS contextualized_transcripts_source_id_fkey;"
            ))
            session.commit()
        return source_id

    @pytest.fixture
    def live_api(self) -> ContextualizerAPI:
        repo = ContextualRepository(db_manager)
        slm_adapter = SLMContextAdapter()
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        return ContextualizerAPI(
            repository=repo,
            orchestrator=orchestrator,
            slm_adapter=slm_adapter
        )

    def test_execute_end_to_end_context_generation(self, isolated_setup: uuid.UUID, live_api: ContextualizerAPI):
        source_id = isolated_setup

        # 1. Create a realistic, structured input transcript (Simulating a medical deposition)
        sentences = [
            ReconstructedSentence(
                sentence_id=uuid.uuid4(), start_sec=0.0, end_sec=5.0,
                text="The patient, John Doe, presented at the clinic on May 4th complaining of chest pain.",
                speaker_id="Speaker_0", words=[]
            ),
            ReconstructedSentence(
                sentence_id=uuid.uuid4(), start_sec=5.0, end_sec=8.0,
                text="He has a history of acute angina and high blood pressure.",
                speaker_id="Speaker_0", words=[]
            ),
            ReconstructedSentence(
                sentence_id=uuid.uuid4(), start_sec=8.0, end_sec=12.0,
                text="I prescribed him 5mg of Amlodipine to be taken daily.",
                speaker_id="Speaker_0", words=[]
            ),
            ReconstructedSentence(
                sentence_id=uuid.uuid4(), start_sec=12.0, end_sec=18.0,
                text="He is scheduled for a follow-up x-ray next week to monitor his aortic valve.",
                speaker_id="Speaker_0", words=[]
            )
        ]

        dummy_transcript = PhoneticCorrectedTranscript(
            source_id=source_id,
            sentences=sentences,
            audit_trail=[]
        )

        try:
            # 2. Execute the Pipeline
            print("\n[DEBUG] Spinning up SLM for Contextual Generation... This may take a few seconds.")
            contextualized_transcript = live_api.execute(
                transcript=dummy_transcript,
                release_after=True
            )
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping test due to insufficient VRAM: {e}")
            raise e

        # 3. Verify Domain Object Structure
        assert isinstance(contextualized_transcript, ContextualizedTranscript)
        assert contextualized_transcript.source_id == source_id

        # Because we only provided 4 sentences (~50 words), the Chunker should group them into exactly 1 chunk
        assert len(contextualized_transcript.chunks) == 1

        chunk = contextualized_transcript.chunks[0]

        # 4. Verify AI Generation (Did the SLM actually read the text and summarize it?)
        assert chunk.context_string is not None
        assert len(chunk.context_string) > 10  # Ensure it's not empty

        # SLM output varies, but it should inherently grasp the medical context
        lower_context = chunk.context_string.lower()
        assert "patient" in lower_context or "john doe" in lower_context or "medical" in lower_context or "doctor" in lower_context

        # Verify the combined payload formatting required by Anthropic's RAG spec
        assert chunk.original_text in chunk.combined_text
        assert chunk.context_string in chunk.combined_text

        # 5. Verify Database Persistence (JSONB Serialization)
        db_transcript = live_api._repo.get_by_source_id(source_id)
        assert db_transcript is not None
        assert db_transcript.id == contextualized_transcript.id
        assert len(db_transcript.chunks) == 1

        # Prove the contextual data survived the trip to Postgres and back
        assert db_transcript.chunks[0].context_string == chunk.context_string
        assert len(db_transcript.chunks[0].sentences) == 4

        print(f"\n[DEBUG] Generated Context: {chunk.context_string}")

    def test_update_transcript_modifies_jsonb_in_db(self, isolated_setup: uuid.UUID, live_api: ContextualizerAPI):
        """Verifies that we can load a transcript, modify its chunk flags, and save it back using update_transcript."""
        source_id = isolated_setup

        # 1. Create and save initial transcript
        chunk = ContextualizedChunk(original_text="Test", needs_visual=False)
        transcript = ContextualizedTranscript(source_id=source_id, chunks=[chunk])
        live_api._repo.save(transcript)

        # 2. Modify the chunk (immutable copy)
        updated_chunk = chunk.model_copy(update={"needs_visual": True})
        updated_transcript = transcript.model_copy(update={"chunks": [updated_chunk]})

        # 3. Call the API update
        result = live_api.update_transcript(updated_transcript)
        assert result.chunks[0].needs_visual is True

        # 4. Fetch directly from the DB to verify the JSONB overwrite was successful
        db_transcript = live_api._repo.get_by_source_id(source_id)
        assert db_transcript.chunks[0].needs_visual is True