# Canvas: test_rag_retrieval_phase_unit_v1
# Path: tests/orchestration/phases/test_rag_retrieval_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock, ANY, patch

from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import RagRetrievalJobResult
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.exceptions.base import ModelManagerError


class TestRagRetrievalPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_retrieval_api(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_loader(self): return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_retrieval_api, mock_async_service, mock_loader):
        return RagRetrievalPhase(
            job_manager=mock_job_manager,
            retrieval_api=mock_retrieval_api,
            async_model_service=mock_async_service,
            loader=mock_loader
        )

    def test_preload_model_triggers_async_service(self, phase, mock_async_service, mock_loader):
        phase.preload_model()
        mock_async_service.trigger_preload.assert_called_once_with(manifest=ANY, loader=mock_loader)

    def test_execute_happy_path(self, phase, mock_async_service, mock_job_manager, mock_loader):
        search_id = uuid.uuid4()
        query = SearchQuery(prompt="Find medical records")
        inputs = [(search_id, query)]

        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future
        mock_loaded_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_loaded_model

        mock_search_result = SearchResult(query="Find medical records", results=[Mock(spec=RetrievedChunk)], execution_time_ms=10)
        mock_job_manager.run_job.return_value = mock_search_result

        result = phase.execute(inputs)

        assert result.phase == PipelinePhase.RAG_RETRIEVAL
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        # Assert job return tuple structure (JobResult, SearchResult)
        job_res, search_res = result.items[0]
        assert isinstance(job_res, RagRetrievalJobResult)
        assert job_res.success is True
        assert job_res.search_id == search_id
        assert search_res == mock_search_result

        mock_job_manager.run_job.assert_called_once_with(ANY, ANY, query=query)
        mock_async_service.trigger_unload.assert_called_once_with(loaded_model=mock_loaded_model, loader=mock_loader, flush=True)

    def test_execute_fails_if_preload_fails(self, phase, mock_async_service):
        phase._preload_future = Mock(spec=concurrent.futures.Future)
        mock_async_service.resolve_preload.side_effect = ModelManagerError("Out of VRAM")

        result = phase.execute([(uuid.uuid4(), SearchQuery(prompt="Test"))])

        assert result.status == PhaseStatus.FAILED
        assert "Critical failure preloading Embedder model" in result.errors[0]
    def test_expand_neighbors_uses_configured_top_k(self, phase, mock_retrieval_api):
        chunks = [Mock(), Mock()]
        mock_retrieval_api.expand_neighbors.return_value = chunks + [Mock()]

        with patch("app.orchestration.service.phases.rag_retrieval_phase.settings") as mock_settings:
            mock_settings.RAG_NEIGHBOR_EXPANSION_TOP_K = 2
            result = phase.expand_neighbors(chunks)

        assert result == mock_retrieval_api.expand_neighbors.return_value
        mock_retrieval_api.expand_neighbors.assert_called_once_with(chunks, 2)

    def test_expand_neighbors_failure_keeps_the_chunks(self, phase, mock_retrieval_api):
        chunks = [Mock()]
        mock_retrieval_api.expand_neighbors.side_effect = RuntimeError("db down")

        assert phase.expand_neighbors(chunks) == chunks
