# Path: app/orchestration/service/pipelines/live_reasoning_pipeline.py

import logging
import uuid
from typing import List, Optional, Tuple, Type, TypeVar

from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
from app.features.qa_reasoning.domain.entities import ReasonerDecision
from app.features.qa_reasoning.domain.enums import ReasonerDecisionType
from app.features.qa_generation.domain.enums import CritiqueVerdict

from app.orchestration.domain.enums import PhaseStatus
from app.orchestration.domain.entities import (
    RagRetrievalJobResult, RagRerankingJobResult, RetrievalScrubbingJobResult, QAGenerationJobResult,
    ReasonerJobResult, PhaseBatchResult, ReasoningIterationTrace, ReasonedChatResponse
)
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError
from app.orchestration.utils.follow_up import answer_question

from app.orchestration.service.phases.reasoner_phase import ReasonerPhase
from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase
from app.orchestration.service.phases.rag_reranking_phase import RagRerankingPhase
from app.orchestration.service.phases.retrieval_scrubber_phase import RetrievalScrubberPhase
from app.orchestration.service.phases.qa_generation_phase import QAGenerationPhase

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _extract_single(result: PhaseBatchResult, job_result_type: Type[_T]):
    for item in result.items:
        if isinstance(item, tuple) and len(item) == 2:
            job_result, payload = item
            if isinstance(job_result, job_result_type):
                return job_result, payload
    return None, None


def _extract_all(result: PhaseBatchResult, job_result_type: Type[_T]) -> List[Tuple[_T, object]]:
    extracted = []
    for item in result.items:
        if isinstance(item, tuple) and len(item) == 2:
            job_result, payload = item
            if isinstance(job_result, job_result_type):
                extracted.append((job_result, payload))
    return extracted


class LiveReasoningPipeline:
    """
    Multi-pass reasoning chat pipeline: always gathers at least one round of real
    evidence before the Reasoner ever judges the question, then loops a reasoning turn against
    the accumulating evidence pool — re-running the existing retrieve -> rerank -> scrub stack
    on REFINE_QUERY — before drafting an answer with immediate self-critique. The first
    looping/branching pipeline in the codebase — every other Live*Pipeline runs its phases
    exactly once, in a fixed sequence.

    The Reasoner never decides blind: ASK_CLARIFICATION/ANSWER_READY/REFINE_QUERY are always
    judged against at least one real search of the raw question, never against an empty
    evidence pool. `max_iterations` bounds the total number of retrieval passes (the mandatory
    first one included), not just REFINE_QUERY-triggered ones.

    Reuses the same RagRetrievalPhase/RagRerankingPhase/RetrievalScrubberPhase/QAGenerationPhase
    instances LiveQAPipeline already uses; nothing about that pipeline changes.
    """

    def __init__(
            self,
            reasoner_phase: ReasonerPhase,
            retrieval_phase: RagRetrievalPhase,
            reranking_phase: RagRerankingPhase,
            scrubber_phase: RetrievalScrubberPhase,
            qa_generation_phase: QAGenerationPhase,
            max_iterations: int = 3,
            max_critique_retries: int = 1,
    ):
        self._reasoner_phase = reasoner_phase
        self._retrieval_phase = retrieval_phase
        self._reranking_phase = reranking_phase
        self._scrubber_phase = scrubber_phase
        self._qa_generation_phase = qa_generation_phase
        self._max_iterations = max_iterations
        self._max_critique_retries = max_critique_retries

    def _discard_pending_preloads(self) -> None:
        """
        Guarantees no background preload outlives this run. Phases clear their own future once
        resolved, so this only touches preloads for phases that never executed (early return,
        halted branch, or an exception) — correct regardless of where in the loop we stopped.
        """
        for phase in (
                self._reasoner_phase, self._retrieval_phase, self._reranking_phase,
                self._scrubber_phase, self._qa_generation_phase
        ):
            try:
                phase.discard_preload()
            except Exception as e:
                logger.error(f"Failed to discard pending preload for {type(phase).__name__}: {e}")

    def _run_retrieval_pass(self, search_id: uuid.UUID, query_text: str, limit: int) -> List[RetrievedChunk]:
        """One retrieve -> rerank -> scrub pass. Shared by the mandatory first pass and every
        REFINE_QUERY pass — both need the exact same three-phase sequence."""
        self._retrieval_phase.preload_model()
        retrieval_result = self._retrieval_phase.execute(
            [(search_id, SearchQuery(prompt=query_text, limit=limit))]
        )
        if retrieval_result.status == PhaseStatus.FAILED:
            raise WorkflowStateError(f"Pipeline halted: Retrieval failed. Errors: {retrieval_result.errors}")
        raw_results = _extract_all(retrieval_result, RagRetrievalJobResult)

        self._reranking_phase.preload_model()
        rerank_result = self._reranking_phase.execute(raw_results, top_n=limit)
        if rerank_result.status == PhaseStatus.FAILED:
            raise WorkflowStateError(f"Pipeline halted: Reranking failed. Errors: {rerank_result.errors}")
        reranked_results = _extract_all(rerank_result, RagRerankingJobResult)

        self._scrubber_phase.preload_model()
        scrub_result = self._scrubber_phase.execute(reranked_results)
        if scrub_result.status == PhaseStatus.FAILED:
            raise WorkflowStateError(f"Pipeline halted: Scrubbing failed. Errors: {scrub_result.errors}")

        chunks: List[RetrievedChunk] = []
        for job_result, survivors in scrub_result.items:
            if isinstance(job_result, RetrievalScrubbingJobResult) and job_result.success and survivors:
                chunks.extend(survivors)
        return self._retrieval_phase.expand_neighbors(chunks)

    def _order_for_answer(
            self,
            search_id: uuid.UUID,
            question: str,
            evidence_pool: List[RetrievedChunk],
            final_decision: Optional[ReasonerDecision],
    ) -> List[RetrievedChunk]:
        """
        Best-first evidence for the answer model, whose context budget keeps only the front of the list.
        The pool mixes passes searched with different (refined) queries, so it is reranked once against the
        question the user actually asked; chunks the Reasoner's final turn linked its claims to go first.
        The Reasoner's prose itself is never shown to the answer model.
        """
        ordered = evidence_pool
        if len(evidence_pool) > 1:
            pool_result = SearchResult(query=question, results=evidence_pool)
            job = RagRetrievalJobResult(search_id=search_id, success=True, retrieved_chunks_count=len(evidence_pool))
            self._reranking_phase.preload_model()
            rerank_result = self._reranking_phase.execute([(job, pool_result)], top_n=len(evidence_pool))
            _, reranked = _extract_single(rerank_result, RagRerankingJobResult)
            if rerank_result.status == PhaseStatus.COMPLETED and reranked is not None:
                ordered = reranked.results
            else:
                logger.warning(f"Reranking the evidence pool failed; keeping gathering order. {rerank_result.errors}")

        linked = {link.chunk_id for link in final_decision.evidence_links} if final_decision else set()
        return [c for c in ordered if c.chunk_id in linked] + [c for c in ordered if c.chunk_id not in linked]

    @staticmethod
    def _merge_new_chunks(
            evidence_pool: List[RetrievedChunk], new_chunks: List[RetrievedChunk]
    ) -> Tuple[List[RetrievedChunk], int]:
        existing_ids = {c.chunk_id for c in evidence_pool}
        merged_new = [c for c in new_chunks if c.chunk_id not in existing_ids]
        return evidence_pool + merged_new, len(merged_new)

    def _reasoner_turn(
            self,
            search_id: uuid.UUID,
            query: SearchQuery,
            evidence_pool: List[RetrievedChunk],
            decision_history: List[ReasonerDecision],
            trace: List[ReasoningIterationTrace],
            iteration_count: int,
            prior_session_context: Optional[str],
            extra_context: Optional[str],
    ) -> ReasonerDecision:
        """One reasoning turn: preload, execute, validate, and record it in decision_history/trace.
        Shared by the normal reasoning loop and the final restricted-choice turn below."""
        self._reasoner_phase.preload_model()
        reasoner_result = self._reasoner_phase.execute([(
            search_id, query.prompt, evidence_pool, decision_history, prior_session_context, extra_context,
        )])
        if reasoner_result.status == PhaseStatus.FAILED:
            raise WorkflowStateError(f"Pipeline halted: Reasoner turn failed. Errors: {reasoner_result.errors}")

        job_result, decision = _extract_single(reasoner_result, ReasonerJobResult)
        if not job_result or not job_result.success or decision is None:
            raise WorkflowStateError("Pipeline halted: Reasoner produced no decision.")

        decision_history.append(decision)
        trace.append(ReasoningIterationTrace(
            iteration=iteration_count,
            decision=decision.decision.value,
            interim_analysis=decision.interim_analysis,
            refined_query=decision.refined_query,
        ))
        return decision

    _FINAL_DECISION_INSTRUCTION = (
        "You have used every available search pass for this question — no further retrieval is possible. "
        "Based only on the evidence already gathered, you must now choose either \"answer_ready\" (answer as "
        "well as you can, and note the remaining gap in your interim_analysis) or \"ask_clarification\" (if you "
        "genuinely cannot proceed without more specific information from the user). Do not choose \"refine_query\" "
        "— there is no search budget left to act on it."
    )

    def execute_reasoning(
            self,
            search_id: uuid.UUID,
            query: SearchQuery,
            prior_session_context: Optional[str] = None,
            seed_chunks: Optional[List[RetrievedChunk]] = None,
    ) -> ReasonedChatResponse:
        """
        prior_session_context is the conversation so far as text. seed_chunks is the evidence earlier turns of the
        conversation showed; it joins the pool after this question's own first search, so it is what an evidence
        budget drops first.
        """
        logger.info(f"========== Starting Live Reasoned Answer Pipeline for Search [{search_id}] ==========")

        evidence_pool: List[RetrievedChunk] = []
        decision_history: List[ReasonerDecision] = []
        trace: List[ReasoningIterationTrace] = []
        iteration_count = 0
        critique_retry_count = 0
        current_query = query.prompt
        # Carries the critique's gap description into the *next* reasoner turn only; prior
        # session context is likewise only meaningful on the true first turn of the call.
        extra_context: Optional[str] = None
        is_first_turn = True
        standalone_query: Optional[str] = None

        def respond(**kwargs) -> ReasonedChatResponse:
            return ReasonedChatResponse(
                standalone_query=standalone_query,
                evidence_chunk_ids=[c.chunk_id for c in evidence_pool],
                **kwargs
            )

        try:
            # ================= Mandatory initial evidence pass =================
            # The Reasoner must never judge confidence blind: always try the raw question
            # against the real data first, so ASK_CLARIFICATION/ANSWER_READY on the very first
            # reasoning turn is already grounded in a real search, not just the question text.
            # A follow-up ("what did they say after that?") is searched as a standalone query; the Reasoner and
            # the answer model still read the user's own words
            current_query = self._reasoner_phase.rewrite_query(search_id, query.prompt, prior_session_context)
            initial_chunks = self._run_retrieval_pass(search_id, current_query, query.limit)
            evidence_pool, _ = self._merge_new_chunks(evidence_pool, initial_chunks)
            evidence_pool, _ = self._merge_new_chunks(evidence_pool, list(seed_chunks or []))
            standalone_query = current_query
            iteration_count = 1

            while True:
                # ================= Reasoning + Retrieval cycle =================
                while True:
                    decision = self._reasoner_turn(
                        search_id, query, evidence_pool, decision_history, trace, iteration_count,
                        prior_session_context if is_first_turn else None, extra_context,
                    )
                    is_first_turn = False
                    extra_context = None

                    if decision.decision == ReasonerDecisionType.ASK_CLARIFICATION:
                        logger.info(f"Reasoned Answer halted for clarification [{search_id}].")
                        return respond(
                            qa_response=None,
                            clarification_question=decision.clarification_question,
                            iteration_trace=trace,
                            total_iterations=iteration_count,
                        )

                    if decision.decision == ReasonerDecisionType.ANSWER_READY:
                        break  # exit the reasoning+retrieval cycle, proceed to generation

                    # REFINE_QUERY
                    if iteration_count >= self._max_iterations:
                        # Out of search budget but the Reasoner still wants more evidence: force one
                        # final, restricted decision (answer_ready or ask_clarification only) rather
                        # than silently generating an answer it just flagged as not yet supported.
                        logger.info(
                            f"Reasoned Answer hit max_iterations={self._max_iterations}; forcing a final decision."
                        )
                        final_decision = self._reasoner_turn(
                            search_id, query, evidence_pool, decision_history, trace, iteration_count + 1,
                            None, self._FINAL_DECISION_INSTRUCTION,
                        )

                        if final_decision.decision == ReasonerDecisionType.ASK_CLARIFICATION:
                            logger.info(f"Reasoned Answer halted for clarification after budget exhausted [{search_id}].")
                            return respond(
                                qa_response=None,
                                clarification_question=final_decision.clarification_question,
                                iteration_trace=trace,
                                total_iterations=iteration_count,
                            )

                        if final_decision.decision == ReasonerDecisionType.ANSWER_READY:
                            break  # proceed to generation

                        # Non-compliant: it still said refine_query despite no budget remaining.
                        # Honor the "ask rather than guess" preference using its own analysis.
                        logger.warning(
                            f"Reasoner ignored the final-decision instruction and chose refine_query again "
                            f"[{search_id}]; falling back to a clarification."
                        )
                        return respond(
                            qa_response=None,
                            clarification_question=(
                                "I couldn't find evidence that specifically answers this after several searches. "
                                f"Could you clarify or narrow down what you're looking for? (What I found so far: "
                                f"{final_decision.interim_analysis})"
                            ),
                            iteration_trace=trace,
                            total_iterations=iteration_count,
                        )

                    current_query = decision.refined_query or current_query
                    iteration_count += 1

                    new_chunks = self._run_retrieval_pass(search_id, current_query, query.limit)
                    evidence_pool, added_count = self._merge_new_chunks(evidence_pool, new_chunks)
                    trace[-1] = trace[-1].model_copy(update={"chunks_added": added_count})

                    # loop back to the top of this while, which preloads and calls the reasoner again

                # ================= Generation + Self-Critique =================
                answer_evidence = self._order_for_answer(
                    search_id, query.prompt, evidence_pool, decision_history[-1] if decision_history else None
                )
                self._qa_generation_phase.preload_model()
                qa_result = self._qa_generation_phase.execute_with_critique(
                    [(search_id, answer_question(query.prompt, standalone_query), answer_evidence)]
                )
                if qa_result.status == PhaseStatus.FAILED:
                    raise WorkflowStateError(
                        f"Pipeline halted: Generation+Critique failed. Errors: {qa_result.errors}"
                    )

                job_result, payload = _extract_single(qa_result, QAGenerationJobResult)
                if not job_result or not job_result.success or payload is None:
                    raise WorkflowStateError("Pipeline halted: No QA response returned from generation+critique.")
                qa_response, critique = payload

                if critique.verdict == CritiqueVerdict.SUFFICIENT:
                    logger.info(f"========== Live Reasoned Answer Pipeline Completed for Search [{search_id}] ==========")
                    return respond(
                        qa_response=qa_response, iteration_trace=trace, total_iterations=iteration_count
                    )

                if critique_retry_count >= self._max_critique_retries:
                    logger.info(f"Reasoned Answer critique unresolved after retry cap for [{search_id}]; returning draft.")
                    return respond(
                        qa_response=qa_response, iteration_trace=trace,
                        total_iterations=iteration_count, critique_unresolved=True,
                    )

                # NEEDS_MORE_EVIDENCE and retries remain: loop back to a reasoner turn with the gap.
                # The outer loop re-enters the inner while, which preloads and calls the reasoner again.
                critique_retry_count += 1
                extra_context = critique.gap_description

        except WorkflowStateError as e:
            logger.error(f"Workflow State Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Critical Reasoned Answer Pipeline Crash: {e}")
            raise WorkflowExecutionError(f"Critical Reasoned Answer Pipeline Crash: {str(e)}", original_exception=e) from e
        finally:
            self._discard_pending_preloads()
