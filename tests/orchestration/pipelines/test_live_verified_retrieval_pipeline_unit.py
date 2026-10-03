# Canvas: test_live_verified_retrieval_pipeline_unit_v2
# Path: tests/orchestration/pipelines/test_live_verified_retrieval_pipeline_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.live_verified_retrieval_pipeline import LiveVerifiedRetrievalPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, RagRetrievalJobResult, RagRerankingJobResult, \
    RetrievalScrubbingJobResult
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
from app.orchestration.exceptions.workflow import WorkflowStateError


class TestLiveVerifiedRetrievalPipelineUnit:
    @pytest.fixture
    def mock_retrieval_phase(self): return Mock()

    @pytest.fixture
    def mock_reranking_phase(self): return Mock()

    @pytest.fixture
    def mock_scrubber_phase(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase):
        return LiveVerifiedRetrievalPipeline(
            retrieval_phase=mock_retrieval_phase,
            reranking_phase=mock_reranking_phase,
            scrubber_phase=mock_scrubber_phase
        )

    def _stub_retrieval_and_rerank(self, search_id, mock_retrieval_phase, mock_reranking_phase, chunk):
        search_result = SearchResult(query="Test Query", results=[chunk], execution_time_ms=10)
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(RagRetrievalJobResult(search_id=search_id, success=True), search_result)]
        )
        rerank_job = RagRerankingJobResult(search_id=search_id, success=True, reranked_chunks_count=1)
        mock_reranking_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(rerank_job, search_result)]
        )
        return rerank_job, search_result

    def test_execute_verified_search_happy_path(self, pipeline, mock_retrieval_phase, mock_reranking_phase,
                                                mock_scrubber_phase):
        search_id = uuid.uuid4()
        query = SearchQuery(prompt="Test Query", limit=4)

        raw_chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Test", start_sec=0,
                                   end_sec=1, relevance_score=1.0)
        rerank_job, reranked_result = self._stub_retrieval_and_rerank(
            search_id, mock_retrieval_phase, mock_reranking_phase, raw_chunk
        )

        scrub_job_result = RetrievalScrubbingJobResult(search_id=search_id, success=True, surviving_chunks_count=1)
        mock_scrubber_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RETRIEVAL_SCRUBBING,
            status=PhaseStatus.COMPLETED,
            successful_count=1,
            items=[(scrub_job_result, [raw_chunk])]
        )

        result = pipeline.execute_verified_search(search_id, query)

        mock_retrieval_phase.preload_model.assert_called_once_with()
        mock_reranking_phase.preload_model.assert_called_once_with()
        # The scrubber loads lazily, only for chunks below the rerank trust score, so it is never preloaded
        mock_scrubber_phase.preload_model.assert_not_called()
        assert mock_reranking_phase.execute.call_args.kwargs == {"top_n": 4}
        mock_scrubber_phase.execute.assert_called_once_with([(rerank_job, reranked_result)])

        assert isinstance(result, SearchResult)
        assert result.query == "Test Query"
        assert result.results == [raw_chunk]

    def test_execute_fails_if_scrubber_fails(self, pipeline, mock_retrieval_phase, mock_reranking_phase,
                                             mock_scrubber_phase):
        search_id = uuid.uuid4()
        raw_chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Test", start_sec=0,
                                   end_sec=1, relevance_score=1.0)
        self._stub_retrieval_and_rerank(search_id, mock_retrieval_phase, mock_reranking_phase, raw_chunk)

        mock_scrubber_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RETRIEVAL_SCRUBBING,
            status=PhaseStatus.FAILED,
            errors=["SLM Crash"]
        )

        with pytest.raises(WorkflowStateError) as exc_info:
            pipeline.execute_verified_search(search_id, SearchQuery(prompt="Test"))

        assert "Pipeline halted: SLM Scrubbing failed" in str(exc_info.value)

    def test_execute_fails_if_reranking_fails_and_discards_scrubber_preload(
            self, pipeline, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase
    ):
        search_id = uuid.uuid4()
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL, status=PhaseStatus.COMPLETED,
            items=[(RagRetrievalJobResult(search_id=search_id, success=True),
                    SearchResult(query="A", results=[], execution_time_ms=0))]
        )
        mock_reranking_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.FAILED, errors=["Reranker crashed"]
        )

        with pytest.raises(WorkflowStateError, match="RAG Reranking failed"):
            pipeline.execute_verified_search(search_id, SearchQuery(prompt="Test"))

        mock_scrubber_phase.execute.assert_not_called()
        mock_scrubber_phase.discard_preload.assert_called_once_with()
