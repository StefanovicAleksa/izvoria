# Canvas: test_live_retrieval_pipeline_unit_v2
# Path: tests/orchestration/pipelines/test_live_retrieval_pipeline_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.live_retrieval_pipeline import LiveRetrievalPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, RagRetrievalJobResult, RagRerankingJobResult
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
from app.orchestration.exceptions.workflow import WorkflowStateError


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=text,
                          start_sec=0, end_sec=1, relevance_score=0.1)


class TestLiveRetrievalPipelineUnit:
    @pytest.fixture
    def mock_retrieval_phase(self): return Mock()

    @pytest.fixture
    def mock_reranking_phase(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_retrieval_phase, mock_reranking_phase):
        return LiveRetrievalPipeline(retrieval_phase=mock_retrieval_phase, reranking_phase=mock_reranking_phase)

    def test_execute_search_retrieves_then_reranks_to_limit(self, pipeline, mock_retrieval_phase, mock_reranking_phase):
        search_id = uuid.uuid4()
        query = SearchQuery(prompt="Test", limit=3)
        candidates = SearchResult(query="Test", results=[_chunk("a"), _chunk("b")], execution_time_ms=10)
        reranked = SearchResult(query="Test", results=[candidates.results[1]], execution_time_ms=10)

        retrieval_job = RagRetrievalJobResult(search_id=search_id, success=True)
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(retrieval_job, candidates)]
        )
        mock_reranking_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(RagRerankingJobResult(search_id=search_id, success=True, reranked_chunks_count=1), reranked)]
        )

        result = pipeline.execute_search(search_id, query)

        mock_retrieval_phase.preload_model.assert_called_once_with()
        mock_reranking_phase.preload_model.assert_called_once_with()
        mock_retrieval_phase.execute.assert_called_once_with([(search_id, query)])
        mock_reranking_phase.execute.assert_called_once_with([(retrieval_job, candidates)], top_n=3)
        assert result == reranked
        mock_retrieval_phase.discard_preload.assert_called_once_with()
        mock_reranking_phase.discard_preload.assert_called_once_with()

    def test_execute_search_fails_if_phase_fails(self, pipeline, mock_retrieval_phase, mock_reranking_phase):
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL,
            status=PhaseStatus.FAILED,
            errors=["OOM Error"]
        )

        with pytest.raises(WorkflowStateError) as exc_info:
            pipeline.execute_search(uuid.uuid4(), SearchQuery(prompt="Test"))

        assert "Pipeline halted: RAG Retrieval failed" in str(exc_info.value)
        mock_reranking_phase.execute.assert_not_called()
        mock_reranking_phase.discard_preload.assert_called_once_with()

    def test_execute_search_fails_if_reranking_fails(self, pipeline, mock_retrieval_phase, mock_reranking_phase):
        search_id = uuid.uuid4()
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL, status=PhaseStatus.COMPLETED,
            items=[(RagRetrievalJobResult(search_id=search_id, success=True),
                    SearchResult(query="Test", results=[_chunk("a")], execution_time_ms=1))]
        )
        mock_reranking_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.FAILED, errors=["Reranker OOM"]
        )

        with pytest.raises(WorkflowStateError, match="RAG Reranking failed"):
            pipeline.execute_search(search_id, SearchQuery(prompt="Test"))
