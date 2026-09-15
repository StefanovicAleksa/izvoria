# Canvas: test_generation_api_integration_v3
# Path: integration_tests/features/qa_generation/test_generation_api_integration.py

import os
import uuid
import pytest
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.qa_generation.data.gemma_adapter import GemmaGenerationAdapter
from app.features.qa_generation.service.generation_api import QAGenerationAPI
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.storage.domain.entities import SourceType

# Import Models to ensure tables are created and for direct DB seeding
from app.features.storage.data.models import SourceModel, FileModel
from app.features.rag_storage.data.models import VectorChunkModel
from app.features.clip_extractor.data.models import VideoClipModel


class TestQAGenerationAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL, NVIDIA GPU, and the Gemma 4 E4B model.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self):
        if not os.path.exists(settings.QA_GENERATION_MODEL_PATH):
            pytest.skip(
                f"Integration test skipped due to missing Gemma 4 model at: {settings.QA_GENERATION_MODEL_PATH}"
            )

    @pytest.fixture
    def live_api(self) -> QAGenerationAPI:
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)
        adapter = GemmaGenerationAdapter()

        return QAGenerationAPI(
            adapter=adapter,
            orchestrator=orchestrator
        )

    @pytest.fixture
    def seeded_db_context(self):
        """Seeds a full relational hierarchy in the live PostgreSQL database."""
        source_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        file_id = uuid.uuid4()
        physical_video_path = "/onyx_data/vault/integration_evidence_clip.mp4"

        with db_manager.session() as session:
            # 1. Bypass foreign key constraints for isolated test seeding
            session.execute(sa_text("ALTER TABLE sources DROP CONSTRAINT IF EXISTS sources_file_id_fkey;"))
            session.execute(
                sa_text("ALTER TABLE vector_chunks DROP CONSTRAINT IF EXISTS vector_chunks_source_id_fkey;"))
            session.execute(sa_text("ALTER TABLE video_clips DROP CONSTRAINT IF EXISTS video_clips_source_id_fkey;"))
            session.execute(
                sa_text("ALTER TABLE video_clips DROP CONSTRAINT IF EXISTS video_clips_extracted_file_id_fkey;"))

            # 2. Seed Physical File Record
            file_model = FileModel(
                id=file_id,
                file_path=physical_video_path,
                file_name="integration_evidence_clip.mp4",
                file_size_bytes=1024576,
                mime_type="video/mp4"
            )
            session.add(file_model)

            # 3. Seed Source Record
            source_model = SourceModel(
                id=source_id,
                name="Bank_Security_Camera.mp4",
                source_type=SourceType.VIDEO.value,
                content_hash=f"hash-{source_id}",
                file_id=file_id
            )
            session.add(source_model)

            # 4. Seed Vector Chunk (The RAG Hit)
            vector_model = VectorChunkModel(
                source_id=source_id,
                chunk_id=chunk_id,
                start_sec=540.0,
                end_sec=550.0,
                text_payload="The defendant wearing a red jacket entered the bank at exactly 09:00 AM.",
                raw_text="[Speaker_0 @ 00:09:00]: The defendant wearing a red jacket entered the bank at exactly 09:00 AM.",
                speaker_labels=["Speaker_0"],
                embedding=[0.01] * 1024
            )
            session.add(vector_model)

            # 5. Seed Extracted Clip Linkage
            clip_model = VideoClipModel(
                source_id=source_id,
                chunk_id=chunk_id,
                extracted_file_id=file_id,
                start_sec=540.0,
                end_sec=550.0
            )
            session.add(clip_model)

            session.commit()

        yield {
            "source_id": source_id,
            "chunk_id": chunk_id,
            "file_id": file_id,
            "physical_path": physical_video_path
        }

        # Cleanup Database
        with db_manager.session() as session:
            session.execute(sa_text("DELETE FROM video_clips WHERE source_id = :sid"), {"sid": str(source_id)})
            session.execute(sa_text("DELETE FROM vector_chunks WHERE source_id = :sid"), {"sid": str(source_id)})
            session.execute(sa_text("DELETE FROM sources WHERE id = :sid"), {"sid": str(source_id)})
            session.execute(sa_text("DELETE FROM files WHERE id = :fid"), {"fid": str(file_id)})
            session.commit()

    def test_end_to_end_qa_generation_with_live_db_and_gpu(self, live_api: QAGenerationAPI, seeded_db_context: dict):
        """
        Tests the entire hardware-bound QA Generation process using the seeded DB relations.
        """
        # 1. Construct the RetrievedChunk exactly as the SearchRepository would return it,
        # resolving the physical_path using the seeded context to simulate the Engine's pipeline transition.
        retrieved_chunk = RetrievedChunk(
            chunk_id=seeded_db_context["chunk_id"],
            source_id=seeded_db_context["source_id"],
            text_payload="The defendant wearing a red jacket entered the bank at exactly 09:00 AM.",
            raw_text="[Speaker_0 @ 00:09:00]: The defendant wearing a red jacket entered the bank at exactly 09:00 AM.",
            speaker_labels=["Speaker_0"],
            start_sec=540.0,
            end_sec=550.0,
            relevance_score=0.95
        )
        # Replicate Engine's path resolution
        object.__setattr__(retrieved_chunk, 'physical_path', seeded_db_context["physical_path"])

        # 2. Execute LIVE GPU Inference
        print("\n[DEBUG] Acquiring VRAM lock and spinning up Gemma 4 E4B for QA Generation...")
        response = live_api.generate(
            user_query="What time did the defendant enter the bank, and what were they wearing?",
            retrieved_chunks=[retrieved_chunk],
            release_after=True
        )

        # 3. Assertions
        assert response.query == "What time did the defendant enter the bank, and what were they wearing?"
        assert response.inference_time_ms > 0

        # Verify the LLM successfully analyzed the text and returned the correct facts
        answer_lower = response.generated_answer.lower()
        assert "09:00 am" in answer_lower or "9:00 am" in answer_lower
        assert "red jacket" in answer_lower

        # Verify the LLM generated the EXACT strict citation format
        assert "[Doc 1]" in response.generated_answer

        # Verify the Citation Parser successfully unpacked the LLM string and linked it to the DB
        assert len(response.citations) == 1
        citation = response.citations[0]

        assert citation.doc_id == "[Doc 1]"
        assert citation.chunk_id == seeded_db_context["chunk_id"]
        assert citation.physical_path == seeded_db_context["physical_path"]
        assert citation.start_sec == 540.0
        assert citation.end_sec == 550.0

    def test_generate_with_critique_end_to_end(self, live_api: QAGenerationAPI, seeded_db_context: dict):
        """Draft + self-critique in one model acquisition, against the same seeded evidence."""
        retrieved_chunk = RetrievedChunk(
            chunk_id=seeded_db_context["chunk_id"],
            source_id=seeded_db_context["source_id"],
            text_payload="The defendant wearing a red jacket entered the bank at exactly 09:00 AM.",
            raw_text="[Speaker_0 @ 00:09:00]: The defendant wearing a red jacket entered the bank at exactly 09:00 AM.",
            speaker_labels=["Speaker_0"],
            start_sec=540.0,
            end_sec=550.0,
            relevance_score=0.95
        )
        object.__setattr__(retrieved_chunk, 'physical_path', seeded_db_context["physical_path"])

        print("\n[DEBUG] Acquiring VRAM lock for Gemma 4 E4B draft + self-critique...")
        response, critique = live_api.generate_with_critique(
            user_query="What time did the defendant enter the bank, and what were they wearing?",
            retrieved_chunks=[retrieved_chunk],
            release_after=True
        )

        assert response.inference_time_ms > 0
        assert "[Doc 1]" in response.generated_answer
        assert critique.verdict is not None