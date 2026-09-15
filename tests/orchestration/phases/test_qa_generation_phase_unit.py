# Canvas: test_qa_generation_phase_unit_v1
# Path: tests/orchestration/phases/test_qa_generation_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock, ANY

from app.orchestration.service.phases.qa_generation_phase import QAGenerationPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import QAGenerationJobResult
from app.features.qa_generation.domain.entities import QAResponse, AnswerCritique
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.exceptions.base import ModelManagerError


class TestQAGenerationPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_qa_api(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_loader(self): return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_qa_api, mock_async_service, mock_loader):
        return QAGenerationPhase(
            job_manager=mock_job_manager,
            qa_api=mock_qa_api,
            async_model_service=mock_async_service,
            loader=mock_loader
        )

    def test_preload_model_triggers_async_service(self, phase, mock_async_service, mock_loader):
        phase.preload_model()
        mock_async_service.trigger_preload.assert_called_once_with(manifest=ANY, loader=mock_loader)

    def test_execute_happy_path(self, phase, mock_async_service, mock_job_manager, mock_loader):
        search_id = uuid.uuid4()
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Test context", start_sec=0,
                               end_sec=1, relevance_score=1.0)
        user_query = "What is the context?"

        inputs = [(search_id, user_query, [chunk])]

        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future
        mock_loaded_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_loaded_model

        mock_qa_response = QAResponse(query=user_query, generated_answer="Test answer", citations=[],
                                      inference_time_ms=10)
        mock_job_manager.run_job.return_value = mock_qa_response

        result = phase.execute(inputs)

        assert result.phase == PipelinePhase.QA_GENERATION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        job_res, qa_res = result.items[0]
        assert isinstance(job_res, QAGenerationJobResult)
        assert job_res.success is True
        assert job_res.search_id == search_id
        assert qa_res == mock_qa_response

        mock_job_manager.run_job.assert_called_once_with(ANY, ANY, user_query=user_query, retrieved_chunks=[chunk],
                                                         release_after=False)
        mock_async_service.trigger_unload.assert_called_once_with(loaded_model=mock_loaded_model, loader=mock_loader,
                                                                  flush=True)

    def test_execute_fails_if_preload_fails(self, phase, mock_async_service):
        phase._preload_future = Mock(spec=concurrent.futures.Future)
        mock_async_service.resolve_preload.side_effect = ModelManagerError("Out of VRAM")

        result = phase.execute([(uuid.uuid4(), "Query", [])])

        assert result.status == PhaseStatus.FAILED
        assert "Critical failure preloading QA Generation LLM" in result.errors[0]

    def test_execute_with_critique_happy_path(self, phase, mock_async_service, mock_job_manager, mock_loader):
        search_id = uuid.uuid4()
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Test context",
                               start_sec=0, end_sec=1, relevance_score=1.0)
        user_query = "What is the context?"

        inputs = [(search_id, user_query, [chunk], "reasoner analysis")]

        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future
        mock_loaded_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_loaded_model

        mock_qa_response = QAResponse(query=user_query, generated_answer="Test answer", citations=[],
                                      inference_time_ms=10)
        mock_critique = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)
        mock_job_manager.run_job.return_value = (mock_qa_response, mock_critique)

        result = phase.execute_with_critique(inputs)

        assert result.phase == PipelinePhase.QA_GENERATION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1

        job_res, payload = result.items[0]
        assert isinstance(job_res, QAGenerationJobResult)
        assert job_res.success is True
        qa_res, critique = payload
        assert qa_res == mock_qa_response
        assert critique == mock_critique

        mock_job_manager.run_job.assert_called_once_with(
            ANY, ANY, user_query=user_query, retrieved_chunks=[chunk], extra_context="reasoner analysis",
            release_after=False
        )
        mock_async_service.trigger_unload.assert_called_once_with(
            loaded_model=mock_loaded_model, loader=mock_loader, flush=True
        )

    def test_execute_with_critique_marks_failure_on_exception(self, phase, mock_job_manager):
        search_id = uuid.uuid4()
        mock_job_manager.run_job.side_effect = RuntimeError("Gemma crashed")

        result = phase.execute_with_critique([(search_id, "Q", [], "")])

        assert result.status == PhaseStatus.FAILED
        job_res, payload = result.items[0]
        assert job_res.success is False
        assert payload is None