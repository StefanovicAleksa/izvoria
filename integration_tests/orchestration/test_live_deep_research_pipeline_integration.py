# Path: integration_tests/orchestration/test_live_deep_research_pipeline_integration.py

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

from app.features.rag_reranking.data.cross_encoder_adapter import CrossEncoderRerankerAdapter
from app.features.rag_reranking.service.rerank_api import RerankAPI
from app.features.retrieval_hallucination_scrubber.data.slm_adapter import QwenRetrievalScrubberAdapter
from app.features.retrieval_hallucination_scrubber.service.scrubber_api import RetrievalScrubberAPI

from app.features.qa_generation.data.gemma_adapter import GemmaGenerationAdapter
from app.features.qa_generation.service.generation_api import QAGenerationAPI

from app.features.qa_reasoning.data.qwen_reasoner_adapter import QwenReasonerAdapter
from app.features.qa_reasoning.service.reasoning_api import ReasoningAPI

from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase
from app.orchestration.service.phases.rag_reranking_phase import RagRerankingPhase
from app.orchestration.service.phases.retrieval_scrubber_phase import RetrievalScrubberPhase
from app.orchestration.service.phases.qa_generation_phase import QAGenerationPhase
from app.orchestration.service.phases.reasoner_phase import ReasonerPhase
from app.orchestration.service.pipelines.live_deep_research_pipeline import LiveDeepResearchPipeline


class TestLiveDeepResearchPipelineIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Tests the LiveDeepResearchPipeline end-to-end, including the Reasoner loop.
    Requires: Live PostgreSQL with pgvector, downloaded Embedding, Scrubber, QA Generation, and
    Reasoner (Qwen2.5-7B) models.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database_and_seed(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)

        required_paths = [
            settings.RAG_EMBEDDING_MODEL_PATH, settings.RETRIEVAL_SCRUBBER_MODEL_PATH,
            settings.QA_GENERATION_MODEL_PATH, settings.QA_REASONER_MODEL_PATH,
        ]
        if any(not os.path.exists(p) for p in required_paths):
            pytest.skip("Integration test skipped: Missing required local models.")

        embedder = HuggingFaceEmbeddingAdapter()
        rag_repo = PostgresRagRepository()

        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        with db_manager.session() as session:
            dummy_file = FileModel(
                id=file_id, file_path="/mock/path/dummy_deep_research.mp4",
                file_name="dummy_deep_research.mp4", file_size_bytes=1024
            )
            dummy_source = SourceModel(
                id=source_id, name="Bank_Security_Camera.mp4", source_type=SourceType.VIDEO.value,
                content_hash="mock_deep_research_hash", file_id=file_id
            )
            session.add(dummy_file)
            session.add(dummy_source)

        texts = [
            "The defendant wearing a red jacket entered the bank at exactly 09:00 AM.",
            "A stray dog wandered into the lobby at 10:15 AM.",
            "The defendant approached the teller and handed over a note at 09:02 AM.",
        ]
        chunks = [
            UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=540.0, end_sec=550.0, text_payload=texts[0]),
            UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=615.0, end_sec=625.0, text_payload=texts[1]),
            UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=542.0, end_sec=552.0, text_payload=texts[2]),
        ]

        manifest = ModelManifest(model_type=ModelType.RAG_EMBEDDING, model_path=settings.RAG_EMBEDDING_MODEL_PATH,
                                 required_vram_mb=100)
        loaded = embedder.load(manifest)
        embeddings = embedder.embed_batch(loaded, texts)
        embedder.unload(loaded)

        rag_repo.replace_rag_chunks(source_id, chunks, embeddings)

        yield

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
                    if type(obj).__name__ in ["Llama", "SentenceTransformer"]:
                        if hasattr(obj, "close"):
                            obj.close()
                except Exception:
                    pass
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

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

        qa_adapter = GemmaGenerationAdapter()
        qa_api = QAGenerationAPI(qa_adapter, orchestrator)
        qa_phase = QAGenerationPhase(job_manager, qa_api, async_service, qa_adapter)

        reranker_adapter = CrossEncoderRerankerAdapter()
        rerank_api = RerankAPI(orchestrator, reranker_adapter)
        rerank_phase = RagRerankingPhase(job_manager, rerank_api, async_service, reranker_adapter)

        reasoner_adapter = QwenReasonerAdapter()
        reasoning_api = ReasoningAPI(adapter=reasoner_adapter, orchestrator=orchestrator)
        reasoner_phase = ReasonerPhase(job_manager, reasoning_api, async_service, reasoner_adapter)

        deep_research_pipeline = LiveDeepResearchPipeline(
            reasoner_phase=reasoner_phase,
            retrieval_phase=rag_phase,
            reranking_phase=rerank_phase,
            scrubber_phase=scrub_phase,
            qa_generation_phase=qa_phase,
            max_iterations=settings.DEEP_RESEARCH_MAX_ITERATIONS,
            max_critique_retries=settings.DEEP_RESEARCH_MAX_CRITIQUE_RETRIES,
        )

        yield deep_research_pipeline
        async_service.shutdown()

    @pytest.mark.integration
    def test_deep_research_pipeline_produces_grounded_answer_or_clarification(self, pipeline_setup):
        deep_research_pipeline = pipeline_setup

        query = SearchQuery(prompt="What time did the defendant enter the bank, and what were they wearing?", limit=3)

        print("\n[DEBUG] Executing End-to-End Deep Research Pipeline (Reasoner + RAG + Chat)...")
        result = deep_research_pipeline.execute_deep_research(uuid.uuid4(), query)

        assert result.total_iterations >= 0
        assert len(result.iteration_trace) >= 1

        if result.qa_response is not None:
            answer_lower = result.qa_response.generated_answer.lower()
            assert "[doc" in answer_lower or "cannot answer" in answer_lower
        else:
            # The Reasoner judged the question ambiguous rather than fabricating an answer.
            assert result.clarification_question
