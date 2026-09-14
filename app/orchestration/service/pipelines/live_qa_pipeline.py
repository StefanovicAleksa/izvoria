# Canvas: live_qa_pipeline_v3
# Path: app/orchestration/service/pipelines/live_qa_pipeline.py

import logging
import uuid
from typing import List, Tuple, Optional

from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult, RetrievedChunk
from app.features.qa_generation.domain.entities import QAResponse

from app.orchestration.domain.enums import PhaseStatus
from app.orchestration.domain.entities import RagRetrievalJobResult, RetrievalScrubbingJobResult, QAGenerationJobResult
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError

from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase
from app.orchestration.service.phases.retrieval_scrubber_phase import RetrievalScrubberPhase
from app.orchestration.service.phases.qa_generation_phase import QAGenerationPhase

logger = logging.getLogger(__name__)


class LiveQAPipeline:
    """
    Synchronous pipeline for executing end-to-end "Chat with your Data".
    Executes Hybrid Search -> SLM Semantic Scrubbing -> Local LLM Final Answer Generation.
    """

    def __init__(
            self,
            retrieval_phase: RagRetrievalPhase,
            scrubber_phase: RetrievalScrubberPhase,
            qa_generation_phase: QAGenerationPhase
    ):
        self._retrieval_phase = retrieval_phase
        self._scrubber_phase = scrubber_phase
        self._qa_generation_phase = qa_generation_phase

    def preload_models(self) -> None:
        """Trigger async loads for all models in the RAG + Chat sequence to warm the VRAM caches."""
        self._retrieval_phase.preload_model()
        self._scrubber_phase.preload_model()
        self._qa_generation_phase.preload_model()

    def _discard_pending_preloads(self) -> None:
        """
        Guarantees no background preload outlives this run. Phases clear their own future
        once resolved, so this only touches preloads for phases that never executed
        (early return, halted branch, or an exception).
        """
        for phase in (self._retrieval_phase, self._scrubber_phase, self._qa_generation_phase):
            try:
                phase.discard_preload()
            except Exception as e:
                logger.error(f"Failed to discard pending preload for {type(phase).__name__}: {e}")

    def execute_chat(self, search_id: uuid.UUID, query: SearchQuery) -> QAResponse:
        logger.info(f"========== Starting Live QA Pipeline for Search [{search_id}] ==========")

        try:
            # ==========================================
            # PRE-WARM & STAGE 1: RAG RETRIEVAL (EMBEDDING)
            # ==========================================
            self.preload_models()

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
            # STAGE 2: RETRIEVAL SCRUBBING (SLM)
            # ==========================================
            scrub_result = self._scrubber_phase.execute(raw_transient_results)

            if scrub_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: SLM Scrubbing failed. Errors: {scrub_result.errors}")

            # Extract surviving pristine chunks
            qa_inputs: List[Tuple[uuid.UUID, str, List[RetrievedChunk]]] = []
            for job_result, surviving_chunks in scrub_result.items:
                if isinstance(job_result, RetrievalScrubbingJobResult) and job_result.success:
                    qa_inputs.append((search_id, query.prompt, surviving_chunks))

            if not qa_inputs:
                raise WorkflowStateError("Pipeline halted: No chunks survived the SLM scrubber.")

            # ==========================================
            # STAGE 3: QA GENERATION (LLM)
            # ==========================================
            qa_phase_result = self._qa_generation_phase.execute(qa_inputs)

            if qa_phase_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: Final LLM Generation failed. Errors: {qa_phase_result.errors}")

            # 4. Extract and Return Final Payload
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
