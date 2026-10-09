# Path: integration_tests/core/model_manager/test_warm_residency_integration.py

import os

import pytest
import torch

from app.core.config import settings
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.features.qa_generation.data.llama_cpp_generation_adapter import LlamaCppGenerationAdapter
from app.features.rag_reranking.data.cross_encoder_adapter import CrossEncoderRerankerAdapter
from app.features.rag_retrieval.data.embedding_adapter import HuggingFacePromptAdapter

pytestmark = pytest.mark.integration


class _CountingLoader:
    """Wraps a real adapter and counts loads and unloads."""

    def __init__(self, inner):
        self._inner = inner
        self.load_count = 0
        self.unload_count = 0

    def load(self, manifest):
        self.load_count += 1
        return self._inner.load(manifest)

    def unload(self, model) -> bool:
        self.unload_count += 1
        return self._inner.unload(model)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TestWarmResidencyIntegration:
    """
    Real BGE-M3, bge-reranker and Qwen3.5-9B (128K) on one card, kept warm the way the bootstrapper wires them.
    Measured on the 12 GB dev card at ~10 GB together; the second chat turn must load nothing.
    """

    @pytest.fixture
    def roles(self):
        if not torch.cuda.is_available():
            pytest.skip("Requires an NVIDIA GPU.")
        roles = [
            (ModelType.RAG_EMBEDDING, settings.RAG_EMBEDDING_MODEL_PATH, settings.RAG_EMBEDDING_VRAM_REQUIREMENT_MB,
             HuggingFacePromptAdapter()),
            (ModelType.RAG_RERANKING, settings.RAG_RERANKER_MODEL_PATH, settings.RAG_RERANKER_VRAM_REQUIREMENT_MB,
             CrossEncoderRerankerAdapter()),
            (ModelType.QA_GENERATION, settings.QA_GENERATION_MODEL_PATH, settings.QA_GENERATION_VRAM_REQUIREMENT_MB,
             LlamaCppGenerationAdapter()),
        ]
        for _, path, _, _ in roles:
            if not os.path.exists(path):
                pytest.skip(f"Model weights missing at {path}.")
        total_mb = torch.cuda.mem_get_info()[1] / 1024 ** 2
        if total_mb < 11000:
            pytest.skip(f"The warm chat set needs a ~12 GB card; this one has {total_mb:.0f} MB.")
        return [(ModelManifest(model_type=t, model_path=p, required_vram_mb=mb), _CountingLoader(a))
                for t, p, mb, a in roles]

    @pytest.fixture
    def services(self):
        orchestrator = ModelManagerOrchestrator(
            PyTorchVramMonitor(),
            keep_warm_types=[ModelType.RAG_EMBEDDING, ModelType.RAG_RERANKING, ModelType.QA_GENERATION]
        )
        async_service = AsyncModelService(BackgroundModelWorker(orchestrator))
        yield orchestrator, async_service
        async_service.release_all()
        async_service.shutdown()

    def test_second_chat_turn_loads_nothing(self, roles, services):
        orchestrator, async_service = services
        (embed_manifest, embedder), (rerank_manifest, reranker), (qa_manifest, qa) = roles

        for _turn in range(2):
            # A live pipeline preloads every phase up front, then resolves each in order
            futures = [async_service.trigger_preload(m, loader) for m, loader in roles]

            embedded = async_service.resolve_preload(futures[0])
            assert len(embedder.embed_prompts(embedded, ["what did the witness say?"])[0]) > 0
            async_service.trigger_unload(embedded, embedder)

            reranking = async_service.resolve_preload(futures[1])
            assert len(reranker.score(reranking, "what did the witness say?", ["the witness said", "snow"])) == 2
            async_service.trigger_unload(reranking, reranker)

            answering = async_service.resolve_preload(futures[2])
            assert answering.model_instance.create_completion("The witness", max_tokens=4)["choices"]
            async_service.trigger_unload(answering, qa)

        # The first turn may evict the embedder/reranker to fit the answer model's conservative first estimate;
        # by the second turn every footprint is measured and all three stay
        loads_after_two_turns = [embedder.load_count, reranker.load_count, qa.load_count]
        assert qa.load_count == 1, f"the answer model was reloaded: loads {loads_after_two_turns}"

        futures = [async_service.trigger_preload(m, loader) for m, loader in roles]
        for future, (_, loader) in zip(futures, roles):
            async_service.trigger_unload(async_service.resolve_preload(future), loader)

        assert [embedder.load_count, reranker.load_count, qa.load_count] == loads_after_two_turns, \
            "a warm turn loaded a model"
        assert qa.unload_count == 0

    def test_release_all_frees_the_card(self, roles, services):
        orchestrator, async_service = services
        monitor = PyTorchVramMonitor()
        monitor.flush_memory()
        free_before = monitor.get_vram_status().free_vram_mb

        for manifest, loader in roles:
            orchestrator.release_model(orchestrator.acquire_model(manifest, loader, wait_for_clear=True), loader)
        assert monitor.get_vram_status().free_vram_mb < free_before - 5000, "the models did not stay resident"

        async_service.release_all()

        assert monitor.get_vram_status().free_vram_mb > free_before - 500

    def test_a_scrub_turn_evicts_the_idle_answer_model_and_it_comes_back(self, roles, services):
        """
        The scrubber does not fit beside the warm set. Acquired synchronously mid-turn, it must evict the IDLE
        answer model (never wait on it), and the answer model's preload after the scrub must load it again.
        """
        from concurrent.futures import ThreadPoolExecutor
        from app.features.retrieval_hallucination_scrubber.data.slm_adapter import \
            QwenRetrievalScrubberAdapter

        if not os.path.exists(settings.RETRIEVAL_SCRUBBER_MODEL_PATH):
            pytest.skip(f"Scrubber weights missing at {settings.RETRIEVAL_SCRUBBER_MODEL_PATH}.")
        orchestrator, async_service = services
        (embed_manifest, embedder), (rerank_manifest, reranker), (qa_manifest, qa) = roles
        scrubber = _CountingLoader(QwenRetrievalScrubberAdapter())
        scrub_manifest = ModelManifest(model_type=ModelType.RETRIEVAL_SCRUBBER,
                                       model_path=settings.RETRIEVAL_SCRUBBER_MODEL_PATH,
                                       required_vram_mb=settings.RETRIEVAL_SCRUBBER_VRAM_REQUIREMENT_MB)
        for manifest, loader in roles:
            orchestrator.release_model(orchestrator.acquire_model(manifest, loader, wait_for_clear=True), loader)

        with ThreadPoolExecutor(max_workers=1) as pool:
            scrubbing = pool.submit(orchestrator.acquire_model, scrub_manifest, scrubber, wait_for_clear=True)
            scrub_model = scrubbing.result(timeout=180)
        orchestrator.release_model(scrub_model, scrubber)

        assert qa.unload_count == 1, "the idle answer model should have made room for the scrubber"
        assert scrubber.unload_count == 1, "the scrubber is not kept warm"

        answering = async_service.resolve_preload(async_service.trigger_preload(qa_manifest, qa))
        assert answering.model_instance.create_completion("The witness", max_tokens=4)["choices"]
        assert qa.load_count == 2
