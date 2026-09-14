# Path: tests/orchestration/phases/test_rag_reranking_phase_unit.py

import uuid
import pytest
from unittest.mock import Mock, ANY

from app.orchestration.service.phases.rag_reranking_phase import RagRerankingPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import RagRetrievalJobResult, RagRerankingJobResult
from app.core.jobs.domain.enums import JobType
from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.rag_retrieval.domain.entities import SearchResult, RetrievedChunk


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=text,
                          start_sec=0, end_sec=1, relevance_score=0.1)


class TestRagRerankingPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_rerank_api(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_loader(self): return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_rerank_api, mock_async_service, mock_loader):
        return RagRerankingPhase(mock_job_manager, mock_rerank_api, mock_async_service, mock_loader)

    def test_preload_skipped_when_reranker_not_installed(self, phase, mock_rerank_api, mock_async_service):
        mock_rerank_api.is_available.return_value = False

        phase.preload_model()

        mock_async_service.trigger_preload.assert_not_called()
        assert phase._preload_future is None

    def test_preload_triggers_with_reranker_manifest(self, phase, mock_rerank_api, mock_async_service, mock_loader):
        mock_rerank_api.is_available.return_value = True

        phase.preload_model()

        mock_async_service.trigger_preload.assert_called_once_with(manifest=mock_rerank_api.manifest, loader=mock_loader)

    def test_execute_reranks_each_successful_retrieval(self, phase, mock_job_manager, mock_rerank_api,
                                                       mock_async_service, mock_loader):
        search_id = uuid.uuid4()
        candidates = [_chunk("a"), _chunk("b")]
        search_result = SearchResult(query="what was said?", results=candidates, execution_time_ms=3)
        retrieval_job = RagRetrievalJobResult(search_id=search_id, success=True)

        future = Mock()
        phase._preload_future = future
        loaded = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = loaded
        mock_job_manager.run_job.return_value = [candidates[1]]

        result = phase.execute([(retrieval_job, search_result)], top_n=1)

        assert result.phase == PipelinePhase.RAG_RERANKING
        assert result.status == PhaseStatus.COMPLETED
        job_result, reranked = result.items[0]
        assert isinstance(job_result, RagRerankingJobResult)
        assert job_result.reranked_chunks_count == 1
        assert reranked.results == [candidates[1]]
        assert reranked.query == "what was said?"

        request = mock_job_manager.run_job.call_args[0][0]
        assert request.job_type == JobType.RAG_RERANKING
        mock_job_manager.run_job.assert_called_once_with(
            ANY, mock_rerank_api.rerank, query="what was said?", chunks=candidates, top_n=1, release_after=False
        )
        mock_async_service.resolve_preload.assert_called_once_with(future)
        mock_async_service.trigger_unload.assert_called_once_with(loaded_model=loaded, loader=mock_loader, flush=True)
        assert phase._preload_future is None

    def test_execute_without_preload_lets_api_release_model(self, phase, mock_job_manager, mock_async_service):
        search_result = SearchResult(query="q", results=[_chunk("a")], execution_time_ms=1)
        mock_job_manager.run_job.return_value = search_result.results

        phase.execute([(RagRetrievalJobResult(search_id=uuid.uuid4(), success=True), search_result)], top_n=5)

        assert mock_job_manager.run_job.call_args.kwargs["release_after"] is True
        mock_async_service.trigger_unload.assert_not_called()

    def test_execute_skips_failed_retrievals(self, phase, mock_job_manager):
        result = phase.execute([(RagRetrievalJobResult(search_id=uuid.uuid4(), success=False), None)], top_n=5)

        mock_job_manager.run_job.assert_not_called()
        assert result.successful_count == 0

    def test_execute_job_failure_marks_phase_failed(self, phase, mock_job_manager):
        search_id = uuid.uuid4()
        mock_job_manager.run_job.side_effect = RuntimeError("Reranker OOM")
        search_result = SearchResult(query="q", results=[_chunk("a")], execution_time_ms=1)

        result = phase.execute([(RagRetrievalJobResult(search_id=search_id, success=True), search_result)], top_n=5)

        assert result.status == PhaseStatus.FAILED
        assert result.items[0] == (
            RagRerankingJobResult(search_id=search_id, success=False, error_message="Reranker OOM"), None
        )

    def test_execute_fails_if_preload_fails(self, phase, mock_async_service):
        phase._preload_future = Mock()
        mock_async_service.resolve_preload.side_effect = ModelManagerError("Out of VRAM")

        result = phase.execute([], top_n=5)

        assert result.status == PhaseStatus.FAILED
        assert "Critical failure preloading Reranker model" in result.errors[0]

    def test_discard_preload_hands_future_to_async_service(self, phase, mock_async_service, mock_loader):
        future = Mock()
        phase._preload_future = future

        phase.discard_preload()

        mock_async_service.discard_preload.assert_called_once_with(future, mock_loader)
        assert phase._preload_future is None
