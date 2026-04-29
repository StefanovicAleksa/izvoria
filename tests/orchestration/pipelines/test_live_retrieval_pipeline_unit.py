# Canvas: test_live_retrieval_pipeline_unit_v1
# Path: tests/orchestration/pipelines/test_live_retrieval_pipeline_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.live_retrieval_pipeline import LiveRetrievalPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, RagRetrievalJobResult
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult
from app.orchestration.exceptions.workflow import WorkflowStateError


class TestLiveRetrievalPipelineUnit:
    @pytest.fixture
    def mock_retrieval_phase(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_retrieval_phase):
        return LiveRetrievalPipeline(retrieval_phase=mock_retrieval_phase)

    def test_execute_search_happy_path(self, pipeline, mock_retrieval_phase):
        search_id = uuid.uuid4()
        query = SearchQuery(prompt="Test")
        search_result = SearchResult(query="Test", results=[], execution_time_ms=10)

        job_result = RagRetrievalJobResult(search_id=search_id, success=True)

        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL,
            status=PhaseStatus.COMPLETED,
            successful_count=1,
            items=[(job_result, search_result)]
        )

        result = pipeline.execute_search(search_id, query)

        mock_retrieval_phase.preload_model.assert_called_once()
        mock_retrieval_phase.execute.assert_called_once()
        assert result == search_result

    def test_execute_search_fails_if_phase_fails(self, pipeline, mock_retrieval_phase):
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL,
            status=PhaseStatus.FAILED,
            errors=["OOM Error"]
        )

        with pytest.raises(WorkflowStateError) as exc_info:
            pipeline.execute_search(uuid.uuid4(), SearchQuery(prompt="Test"))

        assert "Pipeline halted: RAG Retrieval failed" in str(exc_info.value)