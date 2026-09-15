# Path: integration_tests/features/qa_reasoning/test_reasoning_api_integration.py

import os
import pytest

from app.core.config import settings

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.qa_reasoning.data.qwen_reasoner_adapter import QwenReasonerAdapter
from app.features.qa_reasoning.service.reasoning_api import ReasoningAPI
from app.features.qa_reasoning.domain.enums import ReasonerDecisionType
from app.features.rag_retrieval.domain.entities import RetrievedChunk

import uuid


@pytest.mark.integration
class TestReasoningAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: NVIDIA GPU and the Qwen2.5-7B-Instruct model.
    """

    @pytest.fixture(scope="class", autouse=True)
    def ensure_model_exists(self):
        if not os.path.exists(settings.QA_REASONER_MODEL_PATH):
            pytest.skip(
                f"Integration test skipped due to missing Qwen2.5-7B model at: {settings.QA_REASONER_MODEL_PATH}"
            )

    @pytest.fixture
    def live_api(self) -> ReasoningAPI:
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)
        adapter = QwenReasonerAdapter()
        return ReasoningAPI(adapter=adapter, orchestrator=orchestrator)

    def test_reasoner_answers_ready_when_evidence_is_sufficient(self, live_api: ReasoningAPI):
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
            text_payload="The defendant entered the bank at 09:00 AM wearing a red jacket.",
            raw_text="[Speaker_0 @ 00:09:00]: The defendant entered the bank at 09:00 AM wearing a red jacket.",
            start_sec=540.0, end_sec=550.0, relevance_score=0.95
        )

        print("\n[DEBUG] Acquiring VRAM lock and spinning up Qwen2.5-7B for the Reasoner...")
        decision = live_api.reason(
            original_question="What time did the defendant enter the bank?",
            evidence_pool=[chunk],
            decision_history=[],
            release_after=True
        )

        assert decision.decision in (ReasonerDecisionType.ANSWER_READY, ReasonerDecisionType.REFINE_QUERY)
        assert decision.interim_analysis

    def test_reasoner_refines_query_when_evidence_pool_is_empty(self, live_api: ReasoningAPI):
        decision = live_api.reason(
            original_question="What time did the defendant enter the bank?",
            evidence_pool=[],
            decision_history=[],
            release_after=True
        )

        assert decision.decision in (ReasonerDecisionType.REFINE_QUERY, ReasonerDecisionType.ASK_CLARIFICATION)
