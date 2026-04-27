# Canvas: test_rag_retrieval_api_integration_v5
# Path: integration_tests/features/rag_retrieval/test_retrieval_api_integration.py

import os
import uuid
import pytest
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.rag_retrieval.domain.entities import SearchQuery
from app.features.rag_retrieval.data.embedding_adapter import HuggingFacePromptAdapter
from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository
from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI

# Import Models so Base.metadata creates the tables, AND so we can use them for seeding!
from app.features.rag_storage.data.models import VectorChunkModel
from app.features.clip_extractor.data.models import VideoClipModel


class TestRetrievalAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL (with pgvector), NVIDIA GPU, and the bge-m3 model.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self):
        if not os.path.exists(settings.RAG_EMBEDDING_MODEL_PATH):
            pytest.skip(
                f"Integration test skipped due to missing embedding model at: {settings.RAG_EMBEDDING_MODEL_PATH}")

    @pytest.fixture
    def live_api(self) -> RetrievalAPI:
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)
        embedding_adapter = HuggingFacePromptAdapter()
        search_repo = PostgresSearchRepository()

        return RetrievalAPI(
            orchestrator=orchestrator,
            embedding_adapter=embedding_adapter,
            search_repo=search_repo
        )

    @pytest.fixture
    def seeded_data(self):
        """Seeds the database with a known vector and a mock clip linkage using SQLAlchemy ORM."""
        source_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        file_id = uuid.uuid4()

        with db_manager.session() as session:
            # 1. Drop foreign keys for isolated testing (Raw SQL is fine here)
            session.execute(
                sa_text("ALTER TABLE vector_chunks DROP CONSTRAINT IF EXISTS vector_chunks_source_id_fkey;"))
            session.execute(sa_text("ALTER TABLE video_clips DROP CONSTRAINT IF EXISTS video_clips_source_id_fkey;"))
            session.execute(
                sa_text("ALTER TABLE video_clips DROP CONSTRAINT IF EXISTS video_clips_extracted_file_id_fkey;"))

            # 2. Insert the Vector Chunk using ORM (no syntax errors!)
            vector_model = VectorChunkModel(
                source_id=source_id,
                chunk_id=chunk_id,
                start_sec=0.0,
                end_sec=5.0,
                text_payload="The patient presents with severe migraines.",
                embedding=[0.01] * 1024  # pgvector handles the python list automatically
            )
            session.add(vector_model)

            # 3. Insert the VideoClip linkage using ORM
            clip_model = VideoClipModel(
                source_id=source_id,
                chunk_id=chunk_id,
                extracted_file_id=file_id,
                start_sec=0.0,
                end_sec=5.0
            )
            session.add(clip_model)

            session.commit()

        yield {"source_id": source_id, "chunk_id": chunk_id, "extracted_file_id": file_id}

        # Cleanup
        with db_manager.session() as session:
            session.execute(sa_text("DELETE FROM video_clips WHERE source_id = :sid"), {"sid": str(source_id)})
            session.execute(sa_text("DELETE FROM vector_chunks WHERE source_id = :sid"), {"sid": str(source_id)})
            session.commit()

    def test_search_end_to_end(self, live_api: RetrievalAPI, seeded_data: dict):
        query = SearchQuery(
            prompt="patient migraines",
            limit=1,
            source_id_filter=seeded_data["source_id"]
        )

        print("\n[DEBUG] Spinning up BGE-M3 for Prompt Vectorization...")
        result = live_api.search(query)

        # Assertions
        assert result.query == "patient migraines"
        assert len(result.results) == 1
        assert result.execution_time_ms > 0

        top_match = result.results[0]
        assert top_match.chunk_id == seeded_data["chunk_id"]
        assert top_match.source_id == seeded_data["source_id"]
        assert "severe migraines" in top_match.text_payload
        assert top_match.relevance_score > 0.0

        # Verify the LEFT JOIN correctly pulled the Storage ID
        assert top_match.extracted_file_id == seeded_data["extracted_file_id"]