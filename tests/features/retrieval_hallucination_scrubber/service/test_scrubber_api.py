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

    def test_unparseable_verdict_keeps_chunk_and_flags_audit(self, api, mock_adapter, mock_orchestrator):
        from app.features.retrieval_hallucination_scrubber.exceptions.pipeline import JSONScrubParseError

        chunk1 = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Maybe relevant",
                                start_sec=0, end_sec=5, relevance_score=1.0)
        chunk2 = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Off topic",
                                start_sec=5, end_sec=10, relevance_score=0.9)
        mock_adapter.evaluate_relevance.side_effect = [
            JSONScrubParseError("SLM hallucinated invalid JSON"),
            ScrubAudit(chunk_id=chunk2.chunk_id, is_relevant=False, discard_reason="Off topic")
        ]

        surviving, audits = api.scrub_retrieved_chunks("Find info", [chunk1, chunk2], release_after=True)

        assert surviving == [chunk1]
        assert audits[0] == ScrubAudit(chunk_id=chunk1.chunk_id, is_relevant=True, discard_reason=None,
                                       evaluation_failed=True)
        assert audits[1].evaluation_failed is False
        mock_orchestrator.release_model.assert_called_once()

    def test_model_failures_still_abort_the_scrub(self, api, mock_adapter, mock_orchestrator):
        from app.features.retrieval_hallucination_scrubber.exceptions.model import SLMOOMError

        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="x",
                               start_sec=0, end_sec=5, relevance_score=1.0)
        mock_adapter.evaluate_relevance.side_effect = SLMOOMError("VRAM exhausted")

        with pytest.raises(SLMOOMError):
            api.scrub_retrieved_chunks("Find info", [chunk], release_after=True)

        mock_orchestrator.release_model.assert_called_once()


class TestRerankTrust:
    """The SLM rejected chunks the cross-encoder ranked first ("the sign says 'STOP' and not ..."); those skip it."""

    @pytest.fixture
    def adapter(self): return MagicMock()

    @pytest.fixture
    def api(self, adapter): return RetrievalScrubberAPI(adapter=adapter, orchestrator=MagicMock())

    @staticmethod
    def _chunk(text, rerank_score):
        return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=text, start_sec=0,
                              end_sec=5, relevance_score=0.5, rerank_score=rerank_score)

    def test_confidently_reranked_chunk_is_kept_without_asking_the_slm(self, api, adapter, monkeypatch):
        from app.core.config import settings
        monkeypatch.setattr(settings, "RETRIEVAL_SCRUBBER_TRUST_RERANK_SCORE", 0.5)
        sign, weak, unscored = self._chunk("STOP GIMAr MOntaz", 0.93), self._chunk("snow", 0.2), self._chunk("x", None)
        adapter.evaluate_relevance.side_effect = lambda **kw: ScrubAudit(chunk_id=kw["chunk_id"], is_relevant=False,
                                                                         discard_reason="The sign says STOP")

        kept, audits = api.scrub_retrieved_chunks("What does the sign say?", [sign, weak, unscored])

        assert kept == [sign]
        assert [c.kwargs["chunk_id"] for c in adapter.evaluate_relevance.call_args_list] == [weak.chunk_id,
                                                                                             unscored.chunk_id]
        assert audits[0] == ScrubAudit(chunk_id=sign.chunk_id, is_relevant=True, kept_by_rerank_score=True)

    def test_unset_threshold_judges_every_chunk(self, api, adapter, monkeypatch):
        from app.core.config import settings
        monkeypatch.setattr(settings, "RETRIEVAL_SCRUBBER_TRUST_RERANK_SCORE", None)
        chunk = self._chunk("STOP", 0.99)
        adapter.evaluate_relevance.return_value = ScrubAudit(chunk_id=chunk.chunk_id, is_relevant=False)

        kept, _ = api.scrub_retrieved_chunks("What does the sign say?", [chunk])

        assert kept == [] and adapter.evaluate_relevance.call_count == 1

    def test_all_trusted_chunks_never_load_the_slm(self, adapter, monkeypatch):
        """On a warm GPU, loading the scrubber evicts the answer model, so a scrub with nothing to judge loads nothing."""
        from app.core.config import settings
        monkeypatch.setattr(settings, "RETRIEVAL_SCRUBBER_TRUST_RERANK_SCORE", 0.5)
        orchestrator = MagicMock()
        api = RetrievalScrubberAPI(adapter=adapter, orchestrator=orchestrator)
        first, second = self._chunk("a", 0.93), self._chunk("b", 0.5)

        kept, audits = api.scrub_retrieved_chunks("Q?", [first, second])

        assert kept == [first, second]
        assert audits == [ScrubAudit(chunk_id=c.chunk_id, is_relevant=True, kept_by_rerank_score=True)
                          for c in (first, second)]
        orchestrator.acquire_model.assert_not_called()
        orchestrator.release_model.assert_not_called()
        adapter.evaluate_relevance.assert_not_called()

    def test_one_untrusted_chunk_loads_the_slm_once(self, adapter, monkeypatch):
        from app.core.config import settings
        monkeypatch.setattr(settings, "RETRIEVAL_SCRUBBER_TRUST_RERANK_SCORE", 0.5)
        orchestrator = MagicMock()
        api = RetrievalScrubberAPI(adapter=adapter, orchestrator=orchestrator)
        adapter.evaluate_relevance.side_effect = lambda **kw: ScrubAudit(chunk_id=kw["chunk_id"], is_relevant=True)

        api.scrub_retrieved_chunks("Q?", [self._chunk("a", 0.93), self._chunk("b", 0.2)])

        orchestrator.acquire_model.assert_called_once()
        orchestrator.release_model.assert_called_once_with(
            orchestrator.acquire_model.return_value, adapter, flush=True
        )
