# Canvas: test_live_qa_pipeline_unit_v1
# Path: tests/orchestration/pipelines/test_live_qa_pipeline_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.live_qa_pipeline import LiveQAPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, RagRetrievalJobResult, RetrievalScrubbingJobResult, \
    QAGenerationJobResult
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
from app.features.qa_generation.domain.entities import QAResponse
from app.orchestration.exceptions.workflow import WorkflowStateError


class TestLiveQAPipelineUnit:
    @pytest.fixture
    def mock_retrieval_phase(self): return Mock()

    @pytest.fixture
    def mock_scrubber_phase(self): return Mock()

    @pytest.fixture
    def mock_qa_phase(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_retrieval_phase, mock_scrubber_phase, mock_qa_phase):
        return LiveQAPipeline(
            retrieval_phase=mock_retrieval_phase,
            scrubber_phase=mock_scrubber_phase,
            qa_generation_phase=mock_qa_phase
        )

    def test_execute_chat_happy_path(self, pipeline, mock_retrieval_phase, mock_scrubber_phase, mock_qa_phase):
        search_id = uuid.uuid4()
        query = SearchQuery(prompt="Who entered the bank?")

        raw_chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Defendant", start_sec=0,
                                   end_sec=1, relevance_score=1.0)
        search_result = SearchResult(query="Who entered the bank?", results=[raw_chunk], execution_time_ms=10)
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

        qa_response = QAResponse(query="Who entered the bank?", generated_answer="The defendant.", citations=[],
                                 inference_time_ms=50)
        qa_job_result = QAGenerationJobResult(search_id=search_id, success=True, citations_count=0)
        mock_qa_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION,
            status=PhaseStatus.COMPLETED,
            successful_count=1,
            items=[(qa_job_result, qa_response)]
        )

        result = pipeline.execute_chat(search_id, query)

        mock_retrieval_phase.preload_model.assert_called_once()
        mock_scrubber_phase.preload_model.assert_called_once()
        mock_qa_phase.preload_model.assert_called_once()

        mock_retrieval_phase.execute.assert_called_once()
        mock_scrubber_phase.execute.assert_called_once()
        mock_qa_phase.execute.assert_called_once()

        assert isinstance(result, QAResponse)
        assert result.generated_answer == "The defendant."

    def test_execute_fails_if_qa_fails(self, pipeline, mock_retrieval_phase, mock_scrubber_phase, mock_qa_phase):
        retrieval_job_result = RagRetrievalJobResult(search_id=uuid.uuid4(), success=True)
        raw_chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Defendant", start_sec=0,
                                   end_sec=1, relevance_score=1.0)
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL,
            status=PhaseStatus.COMPLETED,
            items=[(retrieval_job_result, SearchResult(query="A", results=[raw_chunk], execution_time_ms=0))]
        )

        scrub_job_result = RetrievalScrubbingJobResult(search_id=uuid.uuid4(), success=True, surviving_chunks_count=1)
        mock_scrubber_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RETRIEVAL_SCRUBBING,
            status=PhaseStatus.COMPLETED,
            items=[(scrub_job_result, [raw_chunk])]
        )

        mock_qa_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION,
            status=PhaseStatus.FAILED,
            errors=["LLM Crash"]
        )

        with pytest.raises(WorkflowStateError) as exc_info:
            pipeline.execute_chat(uuid.uuid4(), SearchQuery(prompt="Test"))

        assert "Pipeline halted: Final LLM Generation failed" in str(exc_info.value)