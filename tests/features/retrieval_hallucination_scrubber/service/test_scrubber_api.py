# Canvas: test_retrieval_scrubber_api_v2
# Path: tests/features/retrieval_hallucination_scrubber/service/test_scrubber_api.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.features.retrieval_hallucination_scrubber.service.scrubber_api import RetrievalScrubberAPI
from app.features.retrieval_hallucination_scrubber.domain.entities import ScrubAudit
from app.features.rag_retrieval.domain.entities import RetrievedChunk


class TestRetrievalScrubberAPI:
    @pytest.fixture
    def mock_adapter(self): return MagicMock()

    @pytest.fixture
    def mock_orchestrator(self): return MagicMock()

    @pytest.fixture
    def api(self, mock_adapter, mock_orchestrator):
        return RetrievalScrubberAPI(
            adapter=mock_adapter,
            orchestrator=mock_orchestrator
        )

    def test_scrub_retrieved_chunks_filters_correctly(self, api, mock_adapter, mock_orchestrator):
        chunk1 = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Relevant info",
                                start_sec=0, end_sec=5, relevance_score=1.0)
        chunk2 = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Garbage info", start_sec=5,
                                end_sec=10, relevance_score=0.9)

        # Mock evaluations: first is relevant, second is not
        mock_adapter.evaluate_relevance.side_effect = [
            ScrubAudit(chunk_id=chunk1.chunk_id, is_relevant=True, discard_reason=None),
            ScrubAudit(chunk_id=chunk2.chunk_id, is_relevant=False, discard_reason="Off topic")
        ]

        surviving_chunks, audits = api.scrub_retrieved_chunks("Find info", [chunk1, chunk2], release_after=True)

        assert len(surviving_chunks) == 1
        assert surviving_chunks[0].chunk_id == chunk1.chunk_id

        assert len(audits) == 2
        assert audits[0].is_relevant is True
        assert audits[1].is_relevant is False

        mock_orchestrator.release_model.assert_called_once()

    def test_scrub_retrieved_chunks_truncates_massive_text(self, api, mock_adapter, mock_orchestrator):
        massive_text = "A" * 5000
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
            text_payload=massive_text, start_sec=0, end_sec=5, relevance_score=1.0
        )

        mock_adapter.evaluate_relevance.return_value = ScrubAudit(
            chunk_id=chunk.chunk_id, is_relevant=True, discard_reason=None
        )

        api.scrub_retrieved_chunks("Find info", [chunk], release_after=True)

        adapter_call_args = mock_adapter.evaluate_relevance.call_args[1]
        passed_text_chunk = adapter_call_args["text_chunk"]

        assert len(passed_text_chunk) == 3000