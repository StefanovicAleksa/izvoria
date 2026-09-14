# Path: integration_tests/features/rag_reranking/test_rerank_api_integration.py

import os
import uuid
import pytest
import torch

from app.core.config import settings
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.rag_reranking.data.cross_encoder_adapter import CrossEncoderRerankerAdapter
from app.features.rag_reranking.service.rerank_api import RerankAPI


@pytest.mark.integration
class TestRerankAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: an NVIDIA GPU and the bge-reranker-v2-m3 weights at RAG_RERANKER_MODEL_PATH.
    """

    @pytest.fixture(scope="module", autouse=True)
    def ensure_hardware_and_weights(self):
        if not os.path.exists(settings.RAG_RERANKER_MODEL_PATH):
            pytest.skip(f"Reranker weights not found at {settings.RAG_RERANKER_MODEL_PATH}; "
                        f"download bge-reranker-v2-m3 there to run this test.")
        if not torch.cuda.is_available():
            pytest.skip("Reranker integration test requires a CUDA device for VRAM orchestration.")

    @pytest.fixture
    def live_api(self) -> RerankAPI:
        orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
        return RerankAPI(orchestrator=orchestrator, adapter=CrossEncoderRerankerAdapter())

    @staticmethod
    def _chunk(text: str, rrf: float) -> RetrievedChunk:
        return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=text,
                              raw_text=text, start_sec=0.0, end_sec=5.0, relevance_score=rrf)

    def test_relevant_chunk_is_promoted_over_higher_rrf_distractor(self, live_api: RerankAPI):
        distractor = self._chunk("[Speaker_1 @ 00:01:00]: The cafeteria serves pizza on Fridays.", rrf=0.033)
        relevant = self._chunk("[Speaker_0 @ 00:04:10]: I prescribed 50mg of sumatriptan for her migraines.", rrf=0.016)

        result = live_api.rerank("What medication was prescribed for the migraines?", [distractor, relevant], top_n=2)

        assert [c.chunk_id for c in result] == [relevant.chunk_id, distractor.chunk_id]
        assert result[0].rerank_score is not None
        assert result[0].rerank_score > result[1].rerank_score
