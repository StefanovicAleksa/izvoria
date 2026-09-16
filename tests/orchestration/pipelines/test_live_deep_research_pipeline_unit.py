# Path: tests/orchestration/pipelines/test_live_deep_research_pipeline_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.live_deep_research_pipeline import LiveDeepResearchPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import (
    PhaseBatchResult, RagRetrievalJobResult, RagRerankingJobResult, RetrievalScrubbingJobResult,
    QAGenerationJobResult, ReasonerJobResult
)
from app.orchestration.exceptions.workflow import WorkflowStateError
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
from app.features.qa_generation.domain.entities import QAResponse, AnswerCritique
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_reasoning.domain.entities import ReasonerDecision, EvidenceLink
from app.features.qa_reasoning.domain.enums import ReasonerDecisionType


def _reasoner_result(search_id, decision: ReasonerDecision, status=PhaseStatus.COMPLETED, errors=None):
    return PhaseBatchResult(
        phase=PipelinePhase.DEEP_RESEARCH_REASONING, status=status, successful_count=1 if status == PhaseStatus.COMPLETED else 0,
        items=[(ReasonerJobResult(search_id=search_id, success=True, decision=decision.decision.value), decision)]
        if status == PhaseStatus.COMPLETED else [],
        errors=errors or []
    )


def _answer_ready(interim="Fully supported.", links=None):
    return ReasonerDecision(
        decision=ReasonerDecisionType.ANSWER_READY, chain_of_thought="cot", interim_analysis=interim,
        evidence_links=links or []
    )


def _refine_query(query: str, interim="Need more."):
    return ReasonerDecision(
        decision=ReasonerDecisionType.REFINE_QUERY, chain_of_thought="cot", interim_analysis=interim,
        refined_query=query
    )


def _ask_clarification(question: str):
    return ReasonerDecision(
        decision=ReasonerDecisionType.ASK_CLARIFICATION, chain_of_thought="cot", interim_analysis="ambiguous",
        clarification_question=question
    )


def _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id,
                          chunks=None):
    """Stubs a single retrieve -> rerank -> scrub pass with `.return_value` (safe for any call
    count) so tests that don't care about evidence content can still satisfy the mandatory
    first pass every LiveDeepResearchPipeline call now performs before the Reasoner runs."""
    chunks = chunks or []
    search_result = SearchResult(query="q", results=chunks, execution_time_ms=1)
    mock_retrieval_phase.execute.return_value = PhaseBatchResult(
        phase=PipelinePhase.RAG_RETRIEVAL, status=PhaseStatus.COMPLETED,
        items=[(RagRetrievalJobResult(search_id=search_id, success=True), search_result)]
    )
    mock_reranking_phase.execute.return_value = PhaseBatchResult(
        phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.COMPLETED,
        items=[(RagRerankingJobResult(search_id=search_id, success=True), search_result)]
    )
    mock_scrubber_phase.execute.return_value = PhaseBatchResult(
        phase=PipelinePhase.RETRIEVAL_SCRUBBING, status=PhaseStatus.COMPLETED,
        items=[(RetrievalScrubbingJobResult(search_id=search_id, success=True), chunks)]
    )


def _echo_rerank(retrieval_results, top_n):
    """Reranking stub that keeps the given order, so tests that don't care about ordering see their input."""
    return PhaseBatchResult(
        phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.COMPLETED,
        items=[(RagRerankingJobResult(search_id=job.search_id, success=True), result)
               for job, result in retrieval_results]
    )


def _sufficient_generation(search_id, answer="Answer."):
    return PhaseBatchResult(
        phase=PipelinePhase.QA_GENERATION, status=PhaseStatus.COMPLETED,
        items=[(QAGenerationJobResult(search_id=search_id, success=True),
                (QAResponse(query="Q", generated_answer=answer, citations=[]),
                 AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)))]
    )


class TestLiveDeepResearchPipelineUnit:
    @pytest.fixture
    def mock_reasoner_phase(self):
        phase = Mock()
        phase.rewrite_query.side_effect = lambda search_id, question, prior: question
        return phase

    @pytest.fixture
    def mock_retrieval_phase(self):
        phase = Mock()
        phase.expand_neighbors.side_effect = lambda chunks: list(chunks)
        return phase

    @pytest.fixture
    def mock_reranking_phase(self):
        phase = Mock()
        phase.execute.side_effect = _echo_rerank
        return phase

    @pytest.fixture
    def mock_scrubber_phase(self): return Mock()

    @pytest.fixture
    def mock_qa_generation_phase(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
                 mock_qa_generation_phase):
        return LiveDeepResearchPipeline(
            reasoner_phase=mock_reasoner_phase,
            retrieval_phase=mock_retrieval_phase,
            reranking_phase=mock_reranking_phase,
            scrubber_phase=mock_scrubber_phase,
            qa_generation_phase=mock_qa_generation_phase,
            max_iterations=3,
            max_critique_retries=1,
        )

    @pytest.fixture
    def raw_chunk(self):
        return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Defendant", start_sec=0,
                              end_sec=1, relevance_score=1.0)

    def _all_phases(self, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
                     mock_qa_generation_phase):
        return (mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
                mock_qa_generation_phase)

    def test_mandatory_first_pass_runs_before_the_reasoner_ever_decides(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            raw_chunk
    ):
        """The Reasoner must never judge blind: its first call must already see evidence from a
        real search, not an empty pool — this is the behavior the whole change exists for."""
        search_id = uuid.uuid4()
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id, [raw_chunk])
        mock_reasoner_phase.execute.return_value = _reasoner_result(search_id, _answer_ready())

        qa_response = QAResponse(query="Q", generated_answer="Answer.", citations=[])
        critique = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)
        pipeline._qa_generation_phase.execute_with_critique.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION, status=PhaseStatus.COMPLETED,
            items=[(QAGenerationJobResult(search_id=search_id, success=True), (qa_response, critique))]
        )

        pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

        mock_retrieval_phase.execute.assert_called_once()
        first_reasoner_call_args = mock_reasoner_phase.execute.call_args_list[0][0][0][0]
        evidence_pool_seen = first_reasoner_call_args[2]
        assert len(evidence_pool_seen) == 1
        assert evidence_pool_seen[0].chunk_id == raw_chunk.chunk_id

    def test_ask_clarification_returns_early_without_generation(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase
    ):
        search_id = uuid.uuid4()
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id)
        mock_reasoner_phase.execute.return_value = _reasoner_result(
            search_id, _ask_clarification("Which incident do you mean?")
        )

        result = pipeline.execute_deep_research(search_id, SearchQuery(prompt="What happened?"))

        assert result.qa_response is None
        assert result.clarification_question == "Which incident do you mean?"
        # total_iterations counts the mandatory first pass even though no REFINE_QUERY fired
        assert result.total_iterations == 1
        mock_qa_generation_phase.execute_with_critique.assert_not_called()

    def test_answer_ready_after_mandatory_pass_does_not_retrieve_again(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase
    ):
        search_id = uuid.uuid4()
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id)
        mock_reasoner_phase.execute.return_value = _reasoner_result(search_id, _answer_ready())

        qa_response = QAResponse(query="Q", generated_answer="Answer.", citations=[])
        critique = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)
        mock_qa_generation_phase.execute_with_critique.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(QAGenerationJobResult(search_id=search_id, success=True), (qa_response, critique))]
        )

        result = pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

        assert result.qa_response == qa_response
        assert result.total_iterations == 1
        # exactly one retrieval pass: the mandatory one. ANSWER_READY triggers no further search.
        mock_retrieval_phase.execute.assert_called_once()

    def test_refine_query_loop_merges_evidence_deduped_by_chunk_id_across_iterations(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase, raw_chunk
    ):
        search_id = uuid.uuid4()
        other_chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Witness",
                                     start_sec=2, end_sec=3, relevance_score=0.9)
        # The mandatory pass surfaces raw_chunk. Reasoner: REFINE_QUERY once (its second pass
        # resurfaces raw_chunk, which must be deduped, alongside a genuinely new chunk), then
        # ANSWER_READY.
        mock_reasoner_phase.execute.side_effect = [
            _reasoner_result(search_id, _refine_query("query two")),
            _reasoner_result(search_id, _answer_ready()),
        ]

        search_result = SearchResult(query="q", results=[raw_chunk], execution_time_ms=1)
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(RagRetrievalJobResult(search_id=search_id, success=True), search_result)]
        )
        mock_reranking_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(RagRerankingJobResult(search_id=search_id, success=True), search_result)]
        )
        # Pass 1 (mandatory) surfaces raw_chunk; pass 2 (REFINE_QUERY) surfaces raw_chunk again
        # (must be deduped) plus other_chunk (new).
        mock_scrubber_phase.execute.side_effect = [
            PhaseBatchResult(
                phase=PipelinePhase.RETRIEVAL_SCRUBBING, status=PhaseStatus.COMPLETED, successful_count=1,
                items=[(RetrievalScrubbingJobResult(search_id=search_id, success=True, surviving_chunks_count=1),
                        [raw_chunk])]
            ),
            PhaseBatchResult(
                phase=PipelinePhase.RETRIEVAL_SCRUBBING, status=PhaseStatus.COMPLETED, successful_count=1,
                items=[(RetrievalScrubbingJobResult(search_id=search_id, success=True, surviving_chunks_count=2),
                        [raw_chunk, other_chunk])]
            ),
        ]

        qa_response = QAResponse(query="Q", generated_answer="Answer.", citations=[])
        critique = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)
        mock_qa_generation_phase.execute_with_critique.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION, status=PhaseStatus.COMPLETED, successful_count=1,
            items=[(QAGenerationJobResult(search_id=search_id, success=True), (qa_response, critique))]
        )

        result = pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q", limit=5))

        assert result.qa_response == qa_response
        assert result.total_iterations == 2
        assert mock_reasoner_phase.execute.call_count == 2
        # generation call receives the deduped evidence pool: raw_chunk once, other_chunk once
        gen_call_args = mock_qa_generation_phase.execute_with_critique.call_args[0][0][0]
        evidence_pool_passed = gen_call_args[2]
        assert len(evidence_pool_passed) == 2
        assert {c.chunk_id for c in evidence_pool_passed} == {raw_chunk.chunk_id, other_chunk.chunk_id}
        # trace[0] = the REFINE_QUERY turn, whose triggered pass added exactly the new chunk
        assert result.iteration_trace[0].chunks_added == 1
        # trace[1] = the ANSWER_READY turn, which triggers no further retrieval
        assert result.iteration_trace[1].chunks_added == 0

    def test_max_iterations_triggers_final_decision_that_answers(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase, raw_chunk
    ):
        search_id = uuid.uuid4()
        # Reasoner wants REFINE_QUERY for all 3 budgeted passes, then complies with the final,
        # budget-exhausted instruction and answers.
        mock_reasoner_phase.execute.side_effect = [
            _reasoner_result(search_id, _refine_query("q1")),
            _reasoner_result(search_id, _refine_query("q2")),
            _reasoner_result(search_id, _refine_query("q3")),
            _reasoner_result(search_id, _answer_ready()),
        ]
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id, [raw_chunk])

        qa_response = QAResponse(query="Q", generated_answer="Best effort answer.", citations=[])
        critique = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)
        mock_qa_generation_phase.execute_with_critique.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION, status=PhaseStatus.COMPLETED,
            items=[(QAGenerationJobResult(search_id=search_id, success=True), (qa_response, critique))]
        )

        result = pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

        assert result.total_iterations == 3
        # 3 REFINE_QUERY turns + 1 final restricted-choice turn = 4 reasoner calls
        assert mock_reasoner_phase.execute.call_count == 4
        # 3 total retrieval passes: the mandatory one plus 2 more REFINE_QUERY-triggered ones
        # (the 3rd REFINE_QUERY hits the cap and is redirected to the final decision instead of retrieving again)
        assert mock_retrieval_phase.execute.call_count == 3
        assert result.qa_response == qa_response
        # the final call was told it has no more search budget
        final_call_args = mock_reasoner_phase.execute.call_args_list[3][0][0][0]
        assert final_call_args[5] == pipeline._FINAL_DECISION_INSTRUCTION

    def test_max_iterations_triggers_final_decision_that_asks_for_clarification(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase, raw_chunk
    ):
        search_id = uuid.uuid4()
        mock_reasoner_phase.execute.side_effect = [
            _reasoner_result(search_id, _refine_query("q1")),
            _reasoner_result(search_id, _refine_query("q2")),
            _reasoner_result(search_id, _refine_query("q3")),
            _reasoner_result(search_id, _ask_clarification("Which financial instrument specifically?")),
        ]
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id, [raw_chunk])

        result = pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

        assert result.qa_response is None
        assert result.clarification_question == "Which financial instrument specifically?"
        mock_qa_generation_phase.execute_with_critique.assert_not_called()

    def test_max_iterations_noncompliant_refine_query_falls_back_to_synthesized_clarification(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase, raw_chunk
    ):
        """If the model ignores the final-decision instruction and insists on refine_query anyway,
        the pipeline must still not silently generate — no more search budget is left, and the
        user prefers being asked over receiving a guess."""
        search_id = uuid.uuid4()
        mock_reasoner_phase.execute.return_value = _reasoner_result(
            search_id, _refine_query("q again", interim="Still not specific enough.")
        )
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id, [raw_chunk])

        result = pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

        assert result.qa_response is None
        assert "Still not specific enough." in result.clarification_question
        assert mock_reasoner_phase.execute.call_count == 4
        mock_qa_generation_phase.execute_with_critique.assert_not_called()

    def test_needs_more_evidence_loops_back_then_respects_retry_cap(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase
    ):
        search_id = uuid.uuid4()
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id)
        mock_reasoner_phase.execute.return_value = _reasoner_result(search_id, _answer_ready())

        qa_response = QAResponse(query="Q", generated_answer="Answer.", citations=[])
        needs_more = AnswerCritique(verdict=CritiqueVerdict.NEEDS_MORE_EVIDENCE, gap_description="Missing X.")
        mock_qa_generation_phase.execute_with_critique.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION, status=PhaseStatus.COMPLETED,
            items=[(QAGenerationJobResult(search_id=search_id, success=True), (qa_response, needs_more))]
        )

        result = pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

        # max_critique_retries=1: one retry attempted (2 reasoner turns, 2 generation calls), then returns anyway
        assert result.critique_unresolved is True
        assert result.qa_response == qa_response
        assert mock_reasoner_phase.execute.call_count == 2
        assert mock_qa_generation_phase.execute_with_critique.call_count == 2

        # the gap description was fed into the second reasoner call as extra_context
        second_call_args = mock_reasoner_phase.execute.call_args_list[1][0][0][0]
        assert second_call_args[5] == "Missing X."

    def test_failed_reasoner_turn_raises_workflow_state_error(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase
    ):
        search_id = uuid.uuid4()
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id)
        mock_reasoner_phase.execute.return_value = _reasoner_result(
            search_id, _answer_ready(), status=PhaseStatus.FAILED, errors=["Reasoner OOM"]
        )

        with pytest.raises(WorkflowStateError, match="Reasoner turn failed"):
            pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

    def test_failed_mandatory_retrieval_raises_before_reasoner_runs_and_discards_all_pending_preloads(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase
    ):
        search_id = uuid.uuid4()
        mock_retrieval_phase.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RETRIEVAL, status=PhaseStatus.FAILED, errors=["Embedder crashed"]
        )

        with pytest.raises(WorkflowStateError, match="Retrieval failed"):
            pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

        # The mandatory pass fails before the Reasoner is ever consulted.
        mock_reasoner_phase.execute.assert_not_called()
        mock_reranking_phase.execute.assert_not_called()
        mock_scrubber_phase.execute.assert_not_called()
        mock_qa_generation_phase.execute_with_critique.assert_not_called()

        for phase in self._all_phases(mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase,
                                      mock_scrubber_phase, mock_qa_generation_phase):
            phase.discard_preload.assert_called_once_with()

    def test_unexpected_crash_still_discards_pending_preloads(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase
    ):
        search_id = uuid.uuid4()
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id)
        mock_reasoner_phase.execute.side_effect = RuntimeError("boom")

        with pytest.raises(Exception):
            pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

        for phase in self._all_phases(mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase,
                                      mock_scrubber_phase, mock_qa_generation_phase):
            phase.discard_preload.assert_called_once_with()

    def test_prior_session_context_only_passed_on_first_turn(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase, raw_chunk
    ):
        search_id = uuid.uuid4()
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id, [raw_chunk])
        mock_reasoner_phase.execute.side_effect = [
            _reasoner_result(search_id, _refine_query("q")),
            _reasoner_result(search_id, _answer_ready()),
        ]
        qa_response = QAResponse(query="Q", generated_answer="Answer.", citations=[])
        critique = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)
        mock_qa_generation_phase.execute_with_critique.return_value = PhaseBatchResult(
            phase=PipelinePhase.QA_GENERATION, status=PhaseStatus.COMPLETED,
            items=[(QAGenerationJobResult(search_id=search_id, success=True), (qa_response, critique))]
        )

        pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"), prior_session_context="prior turn")

        first_call_args = mock_reasoner_phase.execute.call_args_list[0][0][0][0]
        second_call_args = mock_reasoner_phase.execute.call_args_list[1][0][0][0]
        assert first_call_args[4] == "prior turn"
        assert second_call_args[4] is None

    def test_follow_up_is_searched_as_rewritten_but_reasoned_about_as_asked(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase, raw_chunk
    ):
        search_id = uuid.uuid4()
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id, [raw_chunk])
        mock_reasoner_phase.rewrite_query.side_effect = None
        mock_reasoner_phase.rewrite_query.return_value = "Why did the pound reverse after the hike?"
        mock_reasoner_phase.execute.return_value = _reasoner_result(search_id, _answer_ready())
        mock_qa_generation_phase.execute_with_critique.return_value = _sufficient_generation(search_id)

        pipeline.execute_deep_research(search_id, SearchQuery(prompt="Why after that?"), prior_session_context="prior")

        mock_reasoner_phase.rewrite_query.assert_called_once_with(search_id, "Why after that?", "prior")
        searched = mock_retrieval_phase.execute.call_args[0][0][0][1]
        assert searched.prompt == "Why did the pound reverse after the hike?"
        assert mock_reasoner_phase.execute.call_args[0][0][0][1] == "Why after that?"
        assert mock_qa_generation_phase.execute_with_critique.call_args[0][0][0][1] == "Why after that?"

    def test_refined_query_fallback_is_the_rewritten_query(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase
    ):
        search_id = uuid.uuid4()
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id)
        mock_reasoner_phase.rewrite_query.side_effect = None
        mock_reasoner_phase.rewrite_query.return_value = "standalone"
        mock_reasoner_phase.execute.side_effect = [
            _reasoner_result(search_id, _refine_query(None)),
            _reasoner_result(search_id, _answer_ready()),
        ]
        mock_qa_generation_phase.execute_with_critique.return_value = _sufficient_generation(search_id)

        pipeline.execute_deep_research(search_id, SearchQuery(prompt="follow-up"), prior_session_context="prior")

        searched = [call[0][0][0][1].prompt for call in mock_retrieval_phase.execute.call_args_list]
        assert searched == ["standalone", "standalone"]

    def test_each_retrieval_pass_adds_neighbouring_chunks(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase, raw_chunk
    ):
        search_id = uuid.uuid4()
        neighbour = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=raw_chunk.source_id, text_payload="Before",
                                   start_sec=0, end_sec=0.5, relevance_score=0.0)
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id, [raw_chunk])
        mock_retrieval_phase.expand_neighbors.side_effect = lambda chunks: list(chunks) + [neighbour]
        mock_reasoner_phase.execute.return_value = _reasoner_result(search_id, _answer_ready())
        mock_qa_generation_phase.execute_with_critique.return_value = _sufficient_generation(search_id)

        pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

        mock_retrieval_phase.expand_neighbors.assert_called_once_with([raw_chunk])
        pool_seen = mock_reasoner_phase.execute.call_args[0][0][0][2]
        assert [c.chunk_id for c in pool_seen] == [raw_chunk.chunk_id, neighbour.chunk_id]

    def test_answer_evidence_is_reranked_on_the_original_question_with_linked_chunks_first(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase, raw_chunk
    ):
        search_id = uuid.uuid4()
        second = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Second",
                                start_sec=5, end_sec=6, relevance_score=0.5)
        third = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Third",
                               start_sec=7, end_sec=8, relevance_score=0.4)
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id,
                             [raw_chunk, second, third])

        pool_rerank_calls = []

        def rerank(retrieval_results, top_n):
            job, result = retrieval_results[0]
            if result.query == "What did the defendant say?":
                pool_rerank_calls.append((result.results, top_n))
                result = result.model_copy(update={"results": [third, raw_chunk, second]})
            return PhaseBatchResult(
                phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.COMPLETED,
                items=[(RagRerankingJobResult(search_id=job.search_id, success=True), result)]
            )

        mock_reranking_phase.execute.side_effect = rerank
        link = EvidenceLink(chunk_id=second.chunk_id, claim="The defendant spoke.")
        mock_reasoner_phase.execute.return_value = _reasoner_result(
            search_id, _answer_ready(interim="Model-written analysis.", links=[link])
        )
        mock_qa_generation_phase.execute_with_critique.return_value = _sufficient_generation(search_id)

        pipeline.execute_deep_research(search_id, SearchQuery(prompt="What did the defendant say?"))

        assert pool_rerank_calls == [([raw_chunk, second, third], 3)]
        request = mock_qa_generation_phase.execute_with_critique.call_args[0][0][0]
        # Only (search_id, question, evidence): the reasoner's prose is not passed to the answer model
        assert len(request) == 3
        assert [c.chunk_id for c in request[2]] == [second.chunk_id, third.chunk_id, raw_chunk.chunk_id]

    def test_failed_pool_rerank_keeps_gathering_order(
            self, pipeline, mock_reasoner_phase, mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase,
            mock_qa_generation_phase, raw_chunk
    ):
        search_id = uuid.uuid4()
        second = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Second",
                                start_sec=5, end_sec=6, relevance_score=0.5)
        _stub_retrieval_pass(mock_retrieval_phase, mock_reranking_phase, mock_scrubber_phase, search_id,
                             [raw_chunk, second])
        def rerank(retrieval_results, top_n):
            # The retrieval stub's results carry query "q"; the pool rerank uses the user's question "Q"
            if retrieval_results[0][1].query == "Q":
                return PhaseBatchResult(phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.FAILED, errors=["OOM"])
            return _echo_rerank(retrieval_results, top_n)

        mock_reranking_phase.execute.side_effect = rerank
        mock_reasoner_phase.execute.return_value = _reasoner_result(search_id, _answer_ready())
        mock_qa_generation_phase.execute_with_critique.return_value = _sufficient_generation(search_id)

        result = pipeline.execute_deep_research(search_id, SearchQuery(prompt="Q"))

        assert result.qa_response is not None
        request = mock_qa_generation_phase.execute_with_critique.call_args[0][0][0]
        assert [c.chunk_id for c in request[2]] == [raw_chunk.chunk_id, second.chunk_id]
