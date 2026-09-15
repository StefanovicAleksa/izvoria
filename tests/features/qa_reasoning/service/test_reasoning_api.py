# Path: tests/features/qa_reasoning/service/test_reasoning_api.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.features.qa_reasoning.service.reasoning_api import ReasoningAPI
from app.features.qa_reasoning.domain.entities import ReasonerDecision
from app.features.qa_reasoning.domain.enums import ReasonerDecisionType
from app.features.qa_reasoning.exceptions.model import ReasonerOOMError
from app.features.rag_retrieval.domain.entities import RetrievedChunk


class TestReasoningAPI:
    @pytest.fixture
    def mock_adapter(self): return MagicMock()

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
