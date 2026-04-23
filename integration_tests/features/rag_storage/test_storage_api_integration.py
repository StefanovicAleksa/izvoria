# Path: integration_tests/features/rag_storage/test_storage_api_integration.py

# dont remove
import torch
import torchaudio

import os
import uuid
import pytest
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.context_unification.domain.entities import UnifiedTranscript, UnifiedChunk
from app.features.rag_storage.data.postgres_dense_repo import PostgresDenseRepository
from app.features.rag_storage.data.bm25_repo import LocalBM25Repository
from app.features.rag_storage.data.hf_embedder import HuggingFaceEmbeddingAdapter
from app.features.rag_storage.service.storage_api import RAGStorageService


class TestRAGStorageAPIIntegration:
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
    def isolated_setup(self) -> uuid.UUID:
        source_id = uuid.uuid4()
        with db_manager.session() as session:
            session.execute(
                sa_text("ALTER TABLE vector_chunks DROP CONSTRAINT IF EXISTS vector_chunks_source_id_fkey;"))
            session.commit()
        return source_id

    @pytest.fixture
    def live_api(self) -> RAGStorageService:
        dense_repo = PostgresDenseRepository()
        sparse_repo = LocalBM25Repository()
        embedder = HuggingFaceEmbeddingAdapter()

        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        return RAGStorageService(
            dense_repo=dense_repo,
            sparse_repo=sparse_repo,
            orchestrator=orchestrator,
            embedder=embedder
        )

    def test_index_transcript_end_to_end(self, isolated_setup: uuid.UUID, live_api: RAGStorageService):
        source_id = isolated_setup

        chunks = [
            UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0.0, end_sec=5.0,
                         text_payload="The patient presents with severe migraines and nausea."),
            UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=5.0, end_sec=10.0,
                         text_payload="Prescribed 50mg of Sumatriptan to be taken as needed.")
        ]
        transcript = UnifiedTranscript(source_id=source_id, chunks=chunks)

        try:
            print("\n[DEBUG] Spinning up BGE-M3 for Dense Embeddings... This may take a few seconds.")
            result = live_api.index_transcript(source_id, transcript, release_after=True)

            if not result.success:
                pytest.fail(f"Integration failed: {result.error_message}")

        except Exception as e:
            pytest.fail(f"Unhandled system error bypassed the service boundary: {e}")

        # Verify Business Logic
        assert result.success is True
        assert result.telemetry.dense_vectors_saved == 2
        assert result.telemetry.sparse_vectors_saved == 2

        # Verify BM25 Physical File Persistence
        bm25_file = os.path.join(os.getcwd(), "data", "bm25_indexes", f"{source_id}.pkl")
        assert os.path.exists(bm25_file)

        # Verify PostgreSQL Dense Vector Persistence
        with db_manager.session() as session:
            rows = session.execute(
                sa_text("SELECT chunk_id, embedding FROM vector_chunks WHERE source_id = :sid"),
                {"sid": str(source_id)}
            ).fetchall()

            assert len(rows) == 2
            assert rows[0][1] is not None

        # Teardown
        if os.path.exists(bm25_file):
            os.remove(bm25_file)