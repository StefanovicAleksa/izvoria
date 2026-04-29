# Canvas: test_live_verified_retrieval_pipeline_integration_v3
# Path: integration_tests/orchestration/test_live_verified_retrieval_pipeline_integration.py

import os
import gc
import uuid
import pytest

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager
from app.core.jobs.service.manager import JobManager
from app.core.jobs.data.repositories import JobRepository
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.storage.data.models import SourceModel, FileModel
from app.features.storage.domain.entities import SourceType
from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_storage.data.postgres_rag_repo import PostgresRagRepository
from app.features.rag_storage.data.hf_embedder import HuggingFaceEmbeddingAdapter
from app.features.rag_retrieval.domain.entities import SearchQuery
from app.features.rag_retrieval.data.embedding_adapter import HuggingFacePromptAdapter
from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository
from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI

from app.features.retrieval_hallucination_scrubber.data.slm_adapter import QwenRetrievalScrubberAdapter
from app.features.retrieval_hallucination_scrubber.service.scrubber_api import RetrievalScrubberAPI

from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase
from app.orchestration.service.phases.retrieval_scrubber_phase import RetrievalScrubberPhase
from app.orchestration.service.pipelines.live_verified_retrieval_pipeline import LiveVerifiedRetrievalPipeline


class TestLiveVerifiedRetrievalPipelineIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Tests the LiveVerifiedRetrievalPipeline end-to-end.
    Requires: Live PostgreSQL with pgvector, downloaded Embedding & Qwen 3B SLM models.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database_and_seed(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)

        if not os.path.exists(settings.RAG_EMBEDDING_MODEL_PATH) or not os.path.exists(
                settings.RETRIEVAL_SCRUBBER_MODEL_PATH):
            pytest.skip("Integration test skipped: Missing required local models.")

        embedder = HuggingFaceEmbeddingAdapter()
        rag_repo = PostgresRagRepository()

        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        # FIX: Create the prerequisite File and Source parent records
        with db_manager.session() as session:
            dummy_file = FileModel(
                id=file_id,
                file_path="/mock/path/dummy_verified.pdf",
                file_name="dummy_verified.pdf",
                file_size_bytes=1024
            )
            dummy_source = SourceModel(
                id=source_id,
                name="test_verified_inference_seed",
                source_type=SourceType.DOCUMENT.value,
                content_hash="mock_verified_hash",
                file_id=file_id
            )
            session.add(dummy_file)
            session.add(dummy_source)

        texts = [
            "The patient exhibited a severe allergic reaction to penicillin.",
            "The hospital cafeteria serves pizza every Friday.",
            "The legal settlement amount was finalized at three million dollars."
        ]

        chunks = [
            UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0.0, end_sec=5.0, text_payload=texts[0]),
            UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=5.0, end_sec=10.0, text_payload=texts[1]),
            UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=10.0, end_sec=15.0, text_payload=texts[2])
        ]

        manifest = ModelManifest(
            model_type=ModelType.RAG_EMBEDDING,
            model_path=settings.RAG_EMBEDDING_MODEL_PATH,
            required_vram_mb=100
        )
        loaded = embedder.load(manifest)
        embeddings = embedder.embed_batch(loaded, texts)
        embedder.unload(loaded)

        rag_repo.save_rag_chunks(source_id, chunks, embeddings)

        yield

        # Cleanup
        rag_repo.delete_by_source_id(source_id)
        with db_manager.session() as session:
            source = session.get(SourceModel, source_id)
            if source:
                session.delete(source)
            file_rec = session.get(FileModel, file_id)
            if file_rec:
                session.delete(file_rec)

    @pytest.fixture(autouse=True, scope="function")
    def vram_cleanup(self):
        def clear_gpu():
            gc.collect()
            for obj in gc.get_objects():
                try:
                    obj_type = type(obj).__name__
                    if obj_type in ["Llama", "SentenceTransformer"]:
                        if hasattr(obj, "close"):
                            obj.close()
                except Exception:
                    pass
            gc.collect()
            try:
                import torch
            except ImportError:
                torch = None

            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        clear_gpu()
        yield
        clear_gpu()

    @pytest.fixture
    def pipeline_setup(self):
        job_repo = JobRepository(db_manager)
        job_manager = JobManager(job_repo)

        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)
        bg_worker = BackgroundModelWorker(orchestrator)
        async_service = AsyncModelService(bg_worker)

        prompt_adapter = HuggingFacePromptAdapter()
        search_repo = PostgresSearchRepository()
        retrieval_api = RetrievalAPI(orchestrator, prompt_adapter, search_repo)

        rag_phase = RagRetrievalPhase(job_manager, retrieval_api, async_service, prompt_adapter)

        scrubber_adapter = QwenRetrievalScrubberAdapter()
        scrubber_api = RetrievalScrubberAPI(scrubber_adapter, orchestrator)

        scrub_phase = RetrievalScrubberPhase(job_manager, scrubber_api, async_service, scrubber_adapter)

        verified_pipeline = LiveVerifiedRetrievalPipeline(
            retrieval_phase=rag_phase,
            scrubber_phase=scrub_phase
        )

        yield verified_pipeline
        async_service.shutdown()

    @pytest.mark.integration
    def test_verified_live_retrieval_end_to_end(self, pipeline_setup):
        verified_pipeline = pipeline_setup

        broad_query = SearchQuery(prompt="Tell me about the legal settlement amount and any food served.", limit=3)
        broad_result = verified_pipeline.execute_verified_search(uuid.uuid4(), broad_query)

        assert broad_result.query == "Tell me about the legal settlement amount and any food served."
        assert len(broad_result.results) > 0

        strict_query = SearchQuery(
            prompt="Provide exclusively information about medical allergies. Ignore everything else.",
            limit=3
        )
        strict_result = verified_pipeline.execute_verified_search(uuid.uuid4(), strict_query)

        assert len(strict_result.results) >= 1

        for chunk in strict_result.results:
            text = chunk.text_payload.lower()
            assert "penicillin" in text
            assert "pizza" not in text
            assert "million dollars" not in text