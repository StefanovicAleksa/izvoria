# Canvas: test_retrieval_scrubber_api_integration_v3
# Path: integration_tests/features/retrieval_hallucination_scrubber/test_scrubber_api_integration.py

import os
import uuid
import pytest

from app.core.config import settings
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.retrieval_hallucination_scrubber.data.slm_adapter import QwenRetrievalScrubberAdapter
from app.features.retrieval_hallucination_scrubber.service.scrubber_api import RetrievalScrubberAPI


class TestRetrievalScrubberAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Downloaded SLM weights at settings.RETRIEVAL_SCRUBBER_MODEL_PATH.
    """

    @pytest.fixture(scope="module", autouse=True)
    def check_model_downloaded(self):
        if not os.path.exists(settings.RETRIEVAL_SCRUBBER_MODEL_PATH):
            pytest.skip(f"Integration test skipped: Model not found at {settings.RETRIEVAL_SCRUBBER_MODEL_PATH}")

    @pytest.fixture
    def live_api(self) -> RetrievalScrubberAPI:
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)
        adapter = QwenRetrievalScrubberAdapter()

        return RetrievalScrubberAPI(
            adapter=adapter,
            orchestrator=orchestrator
        )

    def test_execute_standard_retrieval_scrubbing(self, live_api: RetrievalScrubberAPI):
        """Verifies real SLM logic accepts relevant RAG context and rejects hallucinations."""

        user_prompt = "What are the indications of the recent MRI scan?"

        chunk_relevant = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            source_id=uuid.uuid4(),
            text_payload="The MRI scan taken on Tuesday shows a minor fracture in the L4 vertebrae.",
            start_sec=10.0,
            end_sec=15.0,
            relevance_score=0.95
        )

        chunk_irrelevant = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            source_id=uuid.uuid4(),
            text_payload="The hospital cafeteria will be serving pizza and salad for lunch on Friday.",
            start_sec=40.0,
            end_sec=45.0,
            relevance_score=0.80
        )

        surviving_chunks, audits = live_api.scrub_retrieved_chunks(
            user_prompt=user_prompt,
            raw_chunks=[chunk_relevant, chunk_irrelevant],
            release_after=True
        )

        assert len(audits) == 2

        # Verify the relevant chunk was kept
        kept_audit = next(a for a in audits if a.chunk_id == chunk_relevant.chunk_id)
        assert kept_audit.is_relevant is True
        assert kept_audit.discard_reason is None

        # Verify the irrelevant chunk was dropped with a reason
        dropped_audit = next(a for a in audits if a.chunk_id == chunk_irrelevant.chunk_id)
        assert dropped_audit.is_relevant is False
        assert dropped_audit.discard_reason is not None

        # Ensure API returned the filtered list
        assert len(surviving_chunks) == 1
        assert surviving_chunks[0].chunk_id == chunk_relevant.chunk_id