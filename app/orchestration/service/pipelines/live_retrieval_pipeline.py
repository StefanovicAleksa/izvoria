# Canvas: live_retrieval_pipeline_v2
# Path: app/orchestration/service/pipelines/live_retrieval_pipeline.py

import logging
import uuid
from typing import List, Optional, Tuple

from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult
from app.orchestration.domain.enums import PhaseStatus
from app.orchestration.domain.entities import RagRetrievalJobResult, RagRerankingJobResult
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError

from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase
from app.orchestration.service.phases.rag_reranking_phase import RagRerankingPhase

logger = logging.getLogger(__name__)


class LiveRetrievalPipeline:
    """
    Synchronous pipeline for executing live user RAG queries.
    Executes Hybrid Search (Dense + Sparse) followed by cross-encoder reranking, without SLM verification.
    """
    def __init__(self, retrieval_phase: RagRetrievalPhase, reranking_phase: RagRerankingPhase):
        self._retrieval_phase = retrieval_phase
        self._reranking_phase = reranking_phase

    def preload_models(self) -> None:
        self._retrieval_phase.preload_model()
        self._reranking_phase.preload_model()

    def _discard_pending_preloads(self) -> None:
        """
        Guarantees no background preload outlives this run. Phases clear their own future
        once resolved, so this only touches preloads for phases that never executed
        (early return, halted branch, or an exception).
        """
        for phase in (self._retrieval_phase, self._reranking_phase):
            try:
                phase.discard_preload()
            except Exception as e:
                logger.error(f"Failed to discard pending preload for {type(phase).__name__}: {e}")

    def execute_search(self, search_id: uuid.UUID, query: SearchQuery) -> SearchResult:
        logger.info(f"========== Starting Live Retrieval Pipeline for Search [{search_id}] ==========")

        try:
            # 1. Warm the Embedder, then the Reranker, in the background
            self.preload_models()

            # 2. Hybrid retrieval of the candidate pool
            queries: List[Tuple[uuid.UUID, SearchQuery]] = [(search_id, query)]
            retrieval_result = self._retrieval_phase.execute(queries)

            if retrieval_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: RAG Retrieval failed. Errors: {retrieval_result.errors}")

            raw_results: List[Tuple[RagRetrievalJobResult, Optional[SearchResult]]] = [
                (job_result, search_result) for job_result, search_result in retrieval_result.items
                if isinstance(job_result, RagRetrievalJobResult)
            ]
            if not raw_results:
                raise WorkflowStateError("Pipeline halted: No search results returned from the database.")

            # 3. Rerank the candidates down to the requested limit
            rerank_result = self._reranking_phase.execute(raw_results, top_n=query.limit)

            if rerank_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: RAG Reranking failed. Errors: {rerank_result.errors}")

            for job_result, search_result in rerank_result.items:
                if isinstance(job_result, RagRerankingJobResult) and job_result.success and search_result:
                    logger.info(f"========== Live Retrieval Pipeline Completed for Search [{search_id}] ==========")
                    return search_result

            raise WorkflowStateError("Pipeline halted: No search results returned from the database.")

        except WorkflowStateError as e:
            logger.error(f"Workflow State Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Critical Pipeline Crash: {e}")
            raise WorkflowExecutionError(f"Critical Pipeline Crash: {str(e)}", original_exception=e) from e
        finally:
            self._discard_pending_preloads()
