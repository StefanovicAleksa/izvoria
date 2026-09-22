# Path: tests/orchestration/phases/test_reasoner_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock, ANY

from app.orchestration.service.phases.reasoner_phase import ReasonerPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import ReasonerJobResult
from app.features.qa_reasoning.domain.entities import ReasonerDecision
from app.features.qa_reasoning.domain.enums import ReasonerDecisionType
from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.exceptions.base import ModelManagerError


class TestReasonerPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_reasoning_api(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_loader(self): return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_reasoning_api, mock_async_service, mock_loader):
        return ReasonerPhase(
            job_manager=mock_job_manager,
            reasoning_api=mock_reasoning_api,
            async_model_service=mock_async_service,
            loader=mock_loader
        )

    def test_preload_model_triggers_async_service(self, phase, mock_async_service, mock_loader):
        phase.preload_model()
        mock_async_service.trigger_preload.assert_called_once_with(manifest=ANY, loader=mock_loader)

    def test_discard_preload_noop_when_nothing_pending(self, phase, mock_async_service):
        phase.discard_preload()
        mock_async_service.discard_preload.assert_not_called()

    def test_discard_preload_releases_pending_future(self, phase, mock_async_service, mock_loader):
        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future

        phase.discard_preload()

        mock_async_service.discard_preload.assert_called_once_with(mock_future, mock_loader)
        assert phase._preload_future is None

    def test_execute_happy_path(self, phase, mock_async_service, mock_job_manager, mock_loader):
        search_id = uuid.uuid4()
        decision = ReasonerDecision(
            decision=ReasonerDecisionType.ANSWER_READY, chain_of_thought="x", interim_analysis="y"
        )
        inputs = [(search_id, "What happened?", [], [], None, None)]

        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future
        mock_loaded_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_loaded_model
        mock_job_manager.run_job.return_value = decision

        result = phase.execute(inputs)

        assert result.phase == PipelinePhase.REASONING_TURN
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        job_res, returned_decision = result.items[0]
        assert isinstance(job_res, ReasonerJobResult)
        assert job_res.success is True
        assert job_res.decision == "answer_ready"
        assert returned_decision == decision

        mock_job_manager.run_job.assert_called_once_with(
            ANY, ANY, original_question="What happened?", evidence_pool=[], decision_history=[],
            prior_session_context=None, extra_context=None, release_after=False
        )
        mock_async_service.trigger_unload.assert_called_once_with(
            loaded_model=mock_loaded_model, loader=mock_loader, flush=True
        )

    def test_execute_fails_if_preload_fails(self, phase, mock_async_service):
        phase._preload_future = Mock(spec=concurrent.futures.Future)
        mock_async_service.resolve_preload.side_effect = ModelManagerError("Out of VRAM")

        result = phase.execute([(uuid.uuid4(), "Q", [], [], None, None)])

        assert result.status == PhaseStatus.FAILED
        assert "Critical failure preloading Reasoner LLM" in result.errors[0]

    def test_execute_marks_failure_on_job_exception(self, phase, mock_job_manager):
        search_id = uuid.uuid4()
        mock_job_manager.run_job.side_effect = RuntimeError("Reasoner crashed")

        result = phase.execute([(search_id, "Q", [], [], None, None)])

        assert result.status == PhaseStatus.FAILED
        assert result.failed_count == 1
        job_res, decision = result.items[0]
        assert job_res.success is False
        assert decision is None

    def test_rewrite_query_without_prior_context_returns_question_without_a_job(self, phase, mock_job_manager):
        assert phase.rewrite_query(uuid.uuid4(), "What happened?", None) == "What happened?"
        mock_job_manager.run_job.assert_not_called()

    def test_rewrite_query_runs_as_tracked_job(self, phase, mock_job_manager, mock_reasoning_api):
        search_id = uuid.uuid4()
        mock_job_manager.run_job.return_value = "standalone query"

        assert phase.rewrite_query(search_id, "And after that?", "prior") == "standalone query"

        request = mock_job_manager.run_job.call_args.args[0]
        assert request.input_params == {"search_id": str(search_id), "step": "query_rewrite"}
        mock_job_manager.run_job.assert_called_once_with(
            ANY, mock_reasoning_api.rewrite_query, question="And after that?", prior_session_context="prior",
            release_after=True
        )

    def test_rewrite_query_failure_falls_back_to_question(self, phase, mock_job_manager):
        mock_job_manager.run_job.side_effect = RuntimeError("Reasoner OOM")

        assert phase.rewrite_query(uuid.uuid4(), "And after that?", "prior") == "And after that?"
