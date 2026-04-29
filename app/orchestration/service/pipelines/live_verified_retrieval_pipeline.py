# Canvas: live_verified_retrieval_pipeline_v2
# Path: app/orchestration/service/pipelines/live_verified_retrieval_pipeline.py

import logging
import uuid
from typing import List, Tuple, Optional

from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult
from app.orchestration.domain.enums import PhaseStatus
from app.orchestration.domain.entities import RagRetrievalJobResult, RetrievalScrubbingJobResult
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError

from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase
from app.orchestration.service.phases.retrieval_scrubber_phase import RetrievalScrubberPhase

logger = logging.getLogger(__name__)


class LiveVerifiedRetrievalPipeline:
    """
    Synchronous pipeline for executing live user RAG queries requiring high precision.
    Executes Hybrid Search, followed immediately by an SLM Hallucination Scrubber to
    drop irrelevant context before returning to the UI/LLM layer.
    """

    def __init__(
            self,
            retrieval_phase: RagRetrievalPhase,
            scrubber_phase: RetrievalScrubberPhase
    ):
        self._retrieval_phase = retrieval_phase
        self._scrubber_phase = scrubber_phase

    def execute_verified_search(self, search_id: uuid.UUID, query: SearchQuery) -> SearchResult:
        logger.info(f"========== Starting Live Verified Retrieval Pipeline for Search [{search_id}] ==========")

        try:
            # ==========================================
            # PHASE 1: RAG RETRIEVAL (EMBEDDING)
            # ==========================================
            # Preload Embedder + Scrubber concurrently to minimize VRAM swap time
            self._retrieval_phase.preload_model()
            self._scrubber_phase.preload_model()

            queries: List[Tuple[uuid.UUID, SearchQuery]] = [(search_id, query)]
            retrieval_phase_result = self._retrieval_phase.execute(queries)

            if retrieval_phase_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: RAG Retrieval failed. Errors: {retrieval_phase_result.errors}")

            # Extract the raw results with explicit typing to satisfy strict linters
            raw_transient_results: List[Tuple[RagRetrievalJobResult, Optional[SearchResult]]] = []
            for item in retrieval_phase_result.items:
                if isinstance(item, tuple) and len(item) == 2:
                    job_result, search_result = item
                    if isinstance(job_result, RagRetrievalJobResult):
                        raw_transient_results.append((job_result, search_result))

            if not raw_transient_results:
                raise WorkflowStateError("Pipeline halted: Failed to extract raw transient results from RAG retrieval.")

            # ==========================================
            # PHASE 2: RETRIEVAL SCRUBBING (SLM)
            # ==========================================
            scrub_phase_result = self._scrubber_phase.execute(raw_transient_results)

            if scrub_phase_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: SLM Scrubbing failed. Errors: {scrub_phase_result.errors}")

            # 3. Construct the final Verified SearchResult
            for job_result, surviving_chunks in scrub_phase_result.items:
                if isinstance(job_result, RetrievalScrubbingJobResult) and job_result.success:
                    verified_result = SearchResult(
                        query=query.prompt,
                        results=surviving_chunks,
                        execution_time_ms=0  # Engine will calculate total latency
                    )

                    logger.info(f"========== Live Verified Retrieval Completed for Search [{search_id}] ==========")
                    return verified_result

            raise WorkflowStateError(f"Pipeline halted: No verified search results returned from the SLM.")

        except WorkflowStateError as e:
            logger.error(f"Workflow State Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Critical Pipeline Crash: {e}")
            raise WorkflowExecutionError(f"Critical Pipeline Crash: {str(e)}", original_exception=e) from e