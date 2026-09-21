# Canvas: live_qa_pipeline_v4
# Path: app/orchestration/service/pipelines/live_qa_pipeline.py

import logging
import uuid
from typing import List, Tuple, Optional

from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
from app.features.qa_generation.domain.entities import QAResponse, ConversationHistory, ConversationAnswer

from app.orchestration.domain.enums import PhaseStatus
from app.orchestration.domain.entities import RagRetrievalJobResult, RagRerankingJobResult, \
    RetrievalScrubbingJobResult, QAGenerationJobResult, ConversationChatResult
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError
from app.orchestration.utils.follow_up import answer_question

from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase
from app.orchestration.service.phases.rag_reranking_phase import RagRerankingPhase
from app.orchestration.service.phases.retrieval_scrubber_phase import RetrievalScrubberPhase
from app.orchestration.service.phases.qa_generation_phase import QAGenerationPhase
from app.orchestration.service.phases.reasoner_phase import ReasonerPhase

logger = logging.getLogger(__name__)


class LiveQAPipeline:
    """
    Synchronous pipeline for executing end-to-end "Chat with your Data".
    Executes Hybrid Search -> Cross-Encoder Reranking -> SLM Semantic Scrubbing -> Local LLM Final Answer Generation.

    execute_conversation_chat() runs the same stages for a question inside a conversation: a follow-up is first
    rewritten into a standalone search query (by the reasoner model), and the answer model sees the earlier turns.
    """

    def __init__(
            self,
            retrieval_phase: RagRetrievalPhase,
            reranking_phase: RagRerankingPhase,
            scrubber_phase: RetrievalScrubberPhase,
            qa_generation_phase: QAGenerationPhase,
            reasoner_phase: Optional[ReasonerPhase] = None
    ):
        self._retrieval_phase = retrieval_phase
        self._reranking_phase = reranking_phase
        self._scrubber_phase = scrubber_phase
        self._qa_generation_phase = qa_generation_phase
        self._reasoner_phase = reasoner_phase

    def preload_models(self) -> None:
        """Trigger async loads for all models in the RAG + Chat sequence to warm the VRAM caches."""
        self._retrieval_phase.preload_model()
        self._reranking_phase.preload_model()
        self._scrubber_phase.preload_model()
        self._qa_generation_phase.preload_model()

    def _discard_pending_preloads(self) -> None:
        """
        Guarantees no background preload outlives this run. Phases clear their own future
        once resolved, so this only touches preloads for phases that never executed
        (early return, halted branch, or an exception).
        """
        phases = [self._retrieval_phase, self._reranking_phase, self._scrubber_phase, self._qa_generation_phase]
        if self._reasoner_phase is not None:
            phases.append(self._reasoner_phase)
        for phase in phases:
            try:
                phase.discard_preload()
            except Exception as e:
                logger.error(f"Failed to discard pending preload for {type(phase).__name__}: {e}")

    def _retrieve_scrubbed(self, search_id: uuid.UUID, query: SearchQuery) -> List[RetrievedChunk]:
        """Stages 1-3: hybrid search -> rerank -> scrub, plus the transcript around the best survivors."""
        # ==========================================
        # STAGE 1: RAG RETRIEVAL (EMBEDDING)
        # ==========================================
        queries: List[Tuple[uuid.UUID, SearchQuery]] = [(search_id, query)]
        retrieval_result = self._retrieval_phase.execute(queries)

        if retrieval_result.status == PhaseStatus.FAILED:
            raise WorkflowStateError(f"Pipeline halted: RAG Retrieval failed. Errors: {retrieval_result.errors}")

        # Extract chunks with explicit inner type validation
        raw_transient_results: List[Tuple[RagRetrievalJobResult, Optional[SearchResult]]] = []
        for item in retrieval_result.items:
            if isinstance(item, tuple) and len(item) == 2:
                job_result, search_result = item
                if isinstance(job_result, RagRetrievalJobResult):
                    raw_transient_results.append((job_result, search_result))

        if not raw_transient_results:
            raise WorkflowStateError("Pipeline halted: Failed to extract SearchResult from Database Phase.")

        # ==========================================
        # STAGE 2: CROSS-ENCODER RERANKING
        # ==========================================
        rerank_result = self._reranking_phase.execute(raw_transient_results, top_n=query.limit)

        if rerank_result.status == PhaseStatus.FAILED:
            raise WorkflowStateError(f"Pipeline halted: RAG Reranking failed. Errors: {rerank_result.errors}")

        reranked_results: List[Tuple[RagRerankingJobResult, Optional[SearchResult]]] = [
            (job_result, search_result) for job_result, search_result in rerank_result.items
            if isinstance(job_result, RagRerankingJobResult)
        ]

        # ==========================================
        # STAGE 3: RETRIEVAL SCRUBBING (SLM)
        # ==========================================
        scrub_result = self._scrubber_phase.execute(reranked_results)

        if scrub_result.status == PhaseStatus.FAILED:
            raise WorkflowStateError(f"Pipeline halted: SLM Scrubbing failed. Errors: {scrub_result.errors}")

        # Extract surviving pristine chunks
        surviving: Optional[List[RetrievedChunk]] = None
        for job_result, surviving_chunks in scrub_result.items:
            if isinstance(job_result, RetrievalScrubbingJobResult) and job_result.success:
                surviving = (surviving or []) + list(surviving_chunks or [])

        if surviving is None:
            raise WorkflowStateError("Pipeline halted: No chunks survived the SLM scrubber.")

        # Transcript on either side of the best survivors, for answers that cross a chunk boundary
        return self._retrieval_phase.expand_neighbors(surviving)

    def execute_chat(self, search_id: uuid.UUID, query: SearchQuery) -> QAResponse:
        logger.info(f"========== Starting Live QA Pipeline for Search [{search_id}] ==========")

        try:
            self.preload_models()
            chunks = self._retrieve_scrubbed(search_id, query)

            if not chunks:
                # Nothing to answer from: generation returns the refusal without a model, so don't keep
                # (or finish) loading one in the background
                self._qa_generation_phase.discard_preload()

            # ==========================================
            # STAGE 4: QA GENERATION (LLM)
            # ==========================================
            qa_phase_result = self._qa_generation_phase.execute([(search_id, query.prompt, chunks)])

            if qa_phase_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: Final LLM Generation failed. Errors: {qa_phase_result.errors}")

            # 5. Extract and Return Final Payload
            for job_result, qa_response in qa_phase_result.items:
                if isinstance(job_result, QAGenerationJobResult) and job_result.success and qa_response:
                    logger.info(f"========== Live QA Pipeline Completed for Search [{search_id}] ==========")
                    return qa_response

            raise WorkflowStateError("Pipeline halted: No QA response returned from the LLM.")

        except WorkflowStateError as e:
            logger.error(f"Workflow State Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Critical QA Pipeline Crash: {e}")
            raise WorkflowExecutionError(f"Critical QA Pipeline Crash: {str(e)}", original_exception=e) from e
        finally:
            self._discard_pending_preloads()

    def execute_conversation_chat(
            self,
            search_id: uuid.UUID,
            query: SearchQuery,
            history: ConversationHistory,
            prior_context: Optional[str] = None
    ) -> ConversationChatResult:
        """
        One turn of a multi-turn chat. prior_context is the earlier turns as plain text, used to rewrite a follow-up
        ("what did he say after that?") into a standalone search query. The answer model reads the earlier turns and
        their evidence, and the user's words followed by that standalone form.
        """
        logger.info(f"========== Starting Conversational QA Pipeline for Search [{search_id}] ==========")

        try:
            search_text = query.prompt
            if prior_context and self._reasoner_phase is not None:
                # Before any preload: the rewrite loads (and releases) its own model, and a queued preload would
                # take the GPU first and never be released while this call waits
                search_text = self._reasoner_phase.rewrite_query(search_id, query.prompt, prior_context)
                logger.info(f"Follow-up searched as: '{search_text}'")

            self.preload_models()
            chunks = self._retrieve_scrubbed(search_id, query.model_copy(update={"prompt": search_text}))

            if not chunks and not any(turn.evidence for turn in history.turns):
                self._qa_generation_phase.discard_preload()

            qa_phase_result = self._qa_generation_phase.execute_conversation(
                [(search_id, answer_question(query.prompt, search_text), chunks, history)]
            )

            if qa_phase_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: Final LLM Generation failed. Errors: {qa_phase_result.errors}")

            for job_result, answer in qa_phase_result.items:
                if isinstance(job_result, QAGenerationJobResult) and job_result.success \
                        and isinstance(answer, ConversationAnswer):
                    logger.info(f"========== Conversational QA Pipeline Completed for Search [{search_id}] ==========")
                    return ConversationChatResult(
                        qa_response=answer.response,
                        standalone_query=search_text,
                        new_chunk_ids=answer.new_chunk_ids
                    )

            raise WorkflowStateError("Pipeline halted: No QA response returned from the LLM.")

        except WorkflowStateError as e:
            logger.error(f"Workflow State Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Critical Conversational QA Pipeline Crash: {e}")
            raise WorkflowExecutionError(
                f"Critical Conversational QA Pipeline Crash: {str(e)}", original_exception=e
            ) from e
        finally:
            self._discard_pending_preloads()
