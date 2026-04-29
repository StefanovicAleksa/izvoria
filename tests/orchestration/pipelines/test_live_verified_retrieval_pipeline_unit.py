# Canvas: test_live_verified_retrieval_pipeline_unit_v1
# Path: tests/orchestration/pipelines/test_live_verified_retrieval_pipeline_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.live_verified_retrieval_pipeline import LiveVerifiedRetrievalPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, RagRetrievalJobResult, RetrievalScrubbingJobResult
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
from app.orchestration.exceptions.workflow import WorkflowStateError


class TestLiveVerifiedRetrievalPipelineUnit:
    @pytest.fixture
    def mock_retrieval_phase(self): return Mock()

    @pytest.fixture
    def mock_scrubber_phase(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_retrieval_phase, mock_scrubber_phase):
        return LiveVerifiedRetrievalPipeline(
            retrieval_phase=mock_retrieval_phase,
            scrubber_phase=mock_scrubber_phase
        )

    def test_execute_verified_search_happy_path(self, pipeline, mock_retrieval_phase, mock_scrubber_phase):
        search_id = uuid.uuid4()
        query = SearchQuery(prompt="Test Query")

        raw_chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Test", start_sec=0,
                                   end_sec=1, relevance_score=1.0)
        search_result = SearchResult(query="Test Query", results=[raw_chunk], execution_time_ms=10)
        retrieval_job_result = RagRetrievalJobResult(search_id=search_id, success=True)

        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL,
            status=PhaseStatus.COMPLETED,
            successful_count=1,
            items=[(retrieval_job_result, search_result)]
        )

        scrub_job_result = RetrievalScrubbingJobResult(search_id=search_id, success=True, surviving_chunks_count=1)
        mock_scrubber_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RETRIEVAL_SCRUBBING,
            status=PhaseStatus.COMPLETED,
            successful_count=1,
            items=[(scrub_job_result, [raw_chunk])]
        )

        result = pipeline.execute_verified_search(search_id, query)

        mock_retrieval_phase.preload_model.assert_called_once()
        mock_scrubber_phase.preload_model.assert_called_once()
        mock_retrieval_phase.execute.assert_called_once()
        mock_scrubber_phase.execute.assert_called_once()

        assert isinstance(result, SearchResult)
        assert result.query == "Test Query"
        assert len(result.results) == 1
        assert result.results[0] == raw_chunk

    def test_execute_fails_if_scrubber_fails(self, pipeline, mock_retrieval_phase, mock_scrubber_phase):
        retrieval_job_result = RagRetrievalJobResult(search_id=uuid.uuid4(), success=True)
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL,
            status=PhaseStatus.COMPLETED,
            items=[(retrieval_job_result, SearchResult(query="A", results=[], execution_time_ms=0))]
        )

        mock_scrubber_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RETRIEVAL_SCRUBBING,
            status=PhaseStatus.FAILED,
            errors=["SLM Crash"]
        )

        with pytest.raises(WorkflowStateError) as exc_info:
            pipeline.execute_verified_search(uuid.uuid4(), SearchQuery(prompt="Test"))

        assert "Pipeline halted: SLM Scrubbing failed" in str(exc_info.value)