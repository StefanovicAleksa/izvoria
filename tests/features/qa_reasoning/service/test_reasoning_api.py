# Path: tests/features/qa_reasoning/service/test_reasoning_api.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.features.qa_reasoning.service.reasoning_api import ReasoningAPI
from app.features.qa_reasoning.domain.entities import ReasonerDecision
from app.features.qa_reasoning.domain.enums import ReasonerDecisionType
from app.features.qa_reasoning.exceptions.model import ReasonerOOMError
from app.features.qa_reasoning.exceptions.pipeline import JSONReasonerParseError
from app.features.rag_retrieval.domain.entities import RetrievedChunk


class TestReasoningAPI:
    @pytest.fixture
    def mock_adapter(self):
        adapter = MagicMock()
        adapter.get_evidence_token_budget.return_value = 5000
        adapter.count_tokens.side_effect = lambda model, text: len(text.split())
        return adapter

    @pytest.fixture
    def mock_orchestrator(self): return MagicMock()

    @pytest.fixture
    def api(self, mock_adapter, mock_orchestrator):
        return ReasoningAPI(adapter=mock_adapter, orchestrator=mock_orchestrator)

    def test_reason_acquires_once_and_releases_once(self, api, mock_adapter, mock_orchestrator):
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="text",
                               start_sec=0, end_sec=1, relevance_score=1.0)
        expected_decision = ReasonerDecision(
            decision=ReasonerDecisionType.ANSWER_READY, chain_of_thought="x", interim_analysis="y"
        )
        mock_adapter.reason.return_value = expected_decision

        decision = api.reason("What happened?", [chunk], [], release_after=True)

        assert decision == expected_decision
        mock_orchestrator.acquire_model.assert_called_once()
        mock_orchestrator.release_model.assert_called_once()

    def test_reason_passes_tagged_evidence_within_budget(self, api, mock_adapter, mock_orchestrator):
        kept = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="short text",
                              start_sec=0, end_sec=1, relevance_score=1.0)
        too_long = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="w " * 100,
                                  start_sec=1, end_sec=2, relevance_score=1.0)
        mock_adapter.get_evidence_token_budget.return_value = 50
        mock_adapter.reason.return_value = ReasonerDecision(
            decision=ReasonerDecisionType.ANSWER_READY, chain_of_thought="x", interim_analysis="y"
        )

        api.reason("What happened?", [kept, too_long], [], prior_session_context="prior", extra_context="gap")

        loaded = mock_orchestrator.acquire_model.return_value
        mock_adapter.get_evidence_token_budget.assert_called_once_with(
            loaded, "What happened?", "(No prior reasoning turns.)", "prior", "gap"
        )
        call_kwargs = mock_adapter.reason.call_args.kwargs
        assert call_kwargs["evidence_block"] == "[E1] (0.0s-1.0s):\nshort text"
        assert call_kwargs["evidence_tags"] == {"E1": kept.chunk_id}

    def test_reason_release_after_false_keeps_model_resident(self, api, mock_adapter, mock_orchestrator):
        mock_adapter.reason.return_value = ReasonerDecision(
            decision=ReasonerDecisionType.ANSWER_READY, chain_of_thought="x", interim_analysis="y"
        )

        api.reason("Q", [], [], release_after=False)

        mock_orchestrator.release_model.assert_not_called()

    def test_reason_releases_even_on_adapter_failure(self, api, mock_adapter, mock_orchestrator):
        mock_adapter.reason.side_effect = ReasonerOOMError("VRAM exhausted")

        with pytest.raises(ReasonerOOMError):
            api.reason("Q", [], [], release_after=True)

        mock_orchestrator.release_model.assert_called_once()

    def test_reason_passes_prior_context_and_extra_context_through(self, api, mock_adapter, mock_orchestrator):
        mock_adapter.reason.return_value = ReasonerDecision(
            decision=ReasonerDecisionType.ANSWER_READY, chain_of_thought="x", interim_analysis="y"
        )

        api.reason("Q", [], [], prior_session_context="prior", extra_context="gap found")

        call_kwargs = mock_adapter.reason.call_args.kwargs
        assert call_kwargs["prior_session_context"] == "prior"
        assert call_kwargs["extra_context"] == "gap found"

    def test_rewrite_query_without_prior_context_skips_the_model(self, api, mock_adapter, mock_orchestrator):
        assert api.rewrite_query("What happened?", None) == "What happened?"

        mock_orchestrator.acquire_model.assert_not_called()
        mock_adapter.rewrite_query.assert_not_called()

    def test_rewrite_query_returns_rewrite_and_releases(self, api, mock_adapter, mock_orchestrator):
        mock_adapter.rewrite_query.return_value = "Why did the pound reverse after the August 2018 hike?"

        result = api.rewrite_query("Why did it reverse?", "[PRIOR TURN] ...")

        assert result == "Why did the pound reverse after the August 2018 hike?"
        mock_adapter.rewrite_query.assert_called_once_with(
            mock_orchestrator.acquire_model.return_value, "Why did it reverse?", "[PRIOR TURN] ..."
        )
        mock_orchestrator.release_model.assert_called_once()

    def test_rewrite_query_falls_back_to_question_on_unparseable_rewrite(self, api, mock_adapter, mock_orchestrator):
        mock_adapter.rewrite_query.side_effect = JSONReasonerParseError("no query")

        assert api.rewrite_query("Why did it reverse?", "prior") == "Why did it reverse?"
        mock_orchestrator.release_model.assert_called_once()

    def test_rewrite_query_raises_typed_error_on_model_failure(self, api, mock_adapter, mock_orchestrator):
        mock_adapter.rewrite_query.side_effect = ReasonerOOMError("VRAM exhausted")

        with pytest.raises(ReasonerOOMError):
            api.rewrite_query("Q", "prior")
        mock_orchestrator.release_model.assert_called_once()
