# Canvas: test_live_qa_pipeline_unit_v2
# Path: tests/orchestration/pipelines/test_live_qa_pipeline_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.live_qa_pipeline import LiveQAPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, RagRetrievalJobResult, RagRerankingJobResult, \
    RetrievalScrubbingJobResult, QAGenerationJobResult
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
from app.features.qa_generation.domain.entities import QAResponse
from app.orchestration.exceptions.workflow import WorkflowStateError


class TestLiveQAPipelineUnit:
    @pytest.fixture
    def mock_retrieval_phase(self):
        phase = Mock()
        phase.expand_neighbors.side_effect = lambda chunks: list(chunks)
        return phase

    @pytest.fixture
    def mock_reranking_phase(self): return Mock()

    @pytest.fixture
    def mock_scrubber_phase(self): return Mock()

    @pytest.fixture
    def mock_qa_phase(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, mock_qa_phase):
        return LiveQAPipeline(
            retrieval_phase=mock_retrieval_phase,
            reranking_phase=mock_reranking_phase,
            scrubber_phase=mock_scrubber_phase,
            qa_generation_phase=mock_qa_phase
        )

    @pytest.fixture
    def raw_chunk(self):
        return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Defendant", start_sec=0,
                              end_sec=1, relevance_score=1.0)

    def _stub_until_scrubber(self, search_id, raw_chunk, mock_retrieval_phase, mock_reranking_phase,
                             mock_scrubber_phase):
        search_result = SearchResult(query="Who entered the bank?", results=[raw_chunk], execution_time_ms=10)
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(RagRetrievalJobResult(search_id=search_id, success=True), search_result)]
        )
        rerank_job = RagRerankingJobResult(search_id=search_id, success=True, reranked_chunks_count=1)
        mock_reranking_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(rerank_job, search_result)]
        )
        mock_scrubber_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RETRIEVAL_SCRUBBING, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(RetrievalScrubbingJobResult(search_id=search_id, success=True, surviving_chunks_count=1),
                    [raw_chunk])]
        )
        return rerank_job, search_result

    def test_execute_chat_happy_path(self, pipeline, mock_retrieval_phase, mock_reranking_phase,
                                     mock_scrubber_phase, mock_qa_phase, raw_chunk):
        search_id = uuid.uuid4()
        query = SearchQuery(prompt="Who entered the bank?", limit=6)
        rerank_job, reranked_result = self._stub_until_scrubber(
            search_id, raw_chunk, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase
        )

        qa_response = QAResponse(query="Who entered the bank?", generated_answer="The defendant.", citations=[],
                                 inference_time_ms=50)
        mock_qa_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION,
            status=PhaseStatus.COMPLETED,
            successful_count=1,
            items=[(QAGenerationJobResult(search_id=search_id, success=True, citations_count=0), qa_response)]
        )

        result = pipeline.execute_chat(search_id, query)

        for phase in (mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, mock_qa_phase):
            phase.preload_model.assert_called_once_with()
            phase.execute.assert_called_once()
            phase.discard_preload.assert_called_once_with()

        assert mock_reranking_phase.execute.call_args.kwargs == {"top_n": 6}
        mock_scrubber_phase.execute.assert_called_once_with([(rerank_job, reranked_result)])
        mock_qa_phase.execute.assert_called_once_with([(search_id, "Who entered the bank?", [raw_chunk])])

        assert isinstance(result, QAResponse)
        assert result.generated_answer == "The defendant."

    def test_execute_fails_if_qa_fails(self, pipeline, mock_retrieval_phase, mock_reranking_phase,
                                       mock_scrubber_phase, mock_qa_phase, raw_chunk):
        self._stub_until_scrubber(uuid.uuid4(), raw_chunk, mock_retrieval_phase, mock_reranking_phase,
                                  mock_scrubber_phase)

        mock_qa_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION,
            status=PhaseStatus.FAILED,
            errors=["LLM Crash"]
        )

        with pytest.raises(WorkflowStateError) as exc_info:
            pipeline.execute_chat(uuid.uuid4(), SearchQuery(prompt="Test"))

        assert "Pipeline halted: Final LLM Generation failed" in str(exc_info.value)

    def test_failed_retrieval_discards_pending_preloads(
            self, pipeline, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, mock_qa_phase
    ):
        """
        Regression: a failed retrieval used to leave the scrubber/QA preloads loading on the
        single worker thread with nobody to release them, hanging every later live call.
        """
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL,
            status=PhaseStatus.FAILED,
            errors=["Embedder crashed"]
        )

        with pytest.raises(WorkflowStateError):
            pipeline.execute_chat(uuid.uuid4(), SearchQuery(prompt="Test"))

        mock_reranking_phase.execute.assert_not_called()
        mock_scrubber_phase.execute.assert_not_called()
        mock_qa_phase.execute.assert_not_called()
        for phase in (mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, mock_qa_phase):
            phase.discard_preload.assert_called_once_with()

    def test_unexpected_crash_still_discards_pending_preloads(
            self, pipeline, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, mock_qa_phase
    ):
        mock_retrieval_phase.execute.side_effect = RuntimeError("boom")

        with pytest.raises(Exception):
            pipeline.execute_chat(uuid.uuid4(), SearchQuery(prompt="Test"))

        for phase in (mock_reranking_phase, mock_scrubber_phase, mock_qa_phase):
            phase.discard_preload.assert_called_once_with()

    def test_failed_reranking_halts_before_scrubbing(self, pipeline, mock_retrieval_phase, mock_reranking_phase,
                                                     mock_scrubber_phase, mock_qa_phase, raw_chunk):
        search_id = uuid.uuid4()
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL, status=PhaseStatus.COMPLETED,
            items=[(RagRetrievalJobResult(search_id=search_id, success=True),
                    SearchResult(query="Q", results=[raw_chunk], execution_time_ms=1))]
        )
        mock_reranking_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.FAILED, errors=["Reranker OOM"]
        )

        with pytest.raises(WorkflowStateError, match="RAG Reranking failed"):
            pipeline.execute_chat(search_id, SearchQuery(prompt="Q"))

        mock_scrubber_phase.execute.assert_not_called()
        mock_qa_phase.execute.assert_not_called()

    def test_answers_from_scrubbed_chunks_plus_their_neighbours(
            self, pipeline, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, mock_qa_phase, raw_chunk
    ):
        search_id = uuid.uuid4()
        neighbour = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=raw_chunk.source_id, text_payload="Before",
                                   start_sec=0, end_sec=0.5, relevance_score=0.0)
        self._stub_until_scrubber(search_id, raw_chunk, mock_retrieval_phase, mock_reranking_phase,
                                  mock_scrubber_phase)
        mock_retrieval_phase.expand_neighbors.side_effect = lambda chunks: list(chunks) + [neighbour]
        qa_response = QAResponse(query="Q", generated_answer="A.", citations=[])
        mock_qa_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(QAGenerationJobResult(search_id=search_id, success=True), qa_response)]
        )

        pipeline.execute_chat(search_id, SearchQuery(prompt="Who entered the bank?"))

        mock_retrieval_phase.expand_neighbors.assert_called_once_with([raw_chunk])
        mock_qa_phase.execute.assert_called_once_with([(search_id, "Who entered the bank?", [raw_chunk, neighbour])])

    def test_no_surviving_chunks_drops_the_answer_model_preload(
            self, pipeline, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, mock_qa_phase, raw_chunk
    ):
        search_id = uuid.uuid4()
        self._stub_until_scrubber(search_id, raw_chunk, mock_retrieval_phase, mock_reranking_phase,
                                  mock_scrubber_phase)
        mock_scrubber_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RETRIEVAL_SCRUBBING, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(RetrievalScrubbingJobResult(search_id=search_id, success=True, surviving_chunks_count=0), [])]
        )
        refusal = QAResponse(query="Q", generated_answer="I cannot answer this based on the provided context.")
        mock_qa_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(QAGenerationJobResult(search_id=search_id, success=True), refusal)]
        )
        call_order = []
        mock_qa_phase.discard_preload.side_effect = lambda: call_order.append("discard")
        mock_qa_phase.execute.side_effect = lambda inputs: call_order.append("execute") or mock_qa_phase.execute.return_value

        result = pipeline.execute_chat(search_id, SearchQuery(prompt="Who entered the bank?"))

        assert result == refusal
        # Dropped before generation runs (and once more by the pipeline's final cleanup)
        assert call_order == ["discard", "execute", "discard"]
        mock_qa_phase.execute.assert_called_once_with([(search_id, "Who entered the bank?", [])])
