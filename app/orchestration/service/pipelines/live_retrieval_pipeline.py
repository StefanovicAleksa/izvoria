# Canvas: live_retrieval_pipeline_v1
# Path: app/orchestration/service/pipelines/live_retrieval_pipeline.py

import logging
import uuid
from typing import List, Tuple

from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult
from app.orchestration.domain.enums import PhaseStatus
from app.orchestration.domain.entities import RagRetrievalJobResult
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError

from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase

logger = logging.getLogger(__name__)


class LiveRetrievalPipeline:
    """
    Synchronous pipeline for executing live user RAG queries.
    Executes standard Hybrid Search (Dense + Sparse) without SLM verification.
    """
    def __init__(self, retrieval_phase: RagRetrievalPhase):
        self._retrieval_phase = retrieval_phase

    def _discard_pending_preloads(self) -> None:
        """
        Guarantees no background preload outlives this run. Phases clear their own future
        once resolved, so this only touches preloads for phases that never executed
        (early return, halted branch, or an exception).
        """
        for phase in (self._retrieval_phase,):
            try:
                phase.discard_preload()
            except Exception as e:
                logger.error(f"Failed to discard pending preload for {type(phase).__name__}: {e}")

    def execute_search(self, search_id: uuid.UUID, query: SearchQuery) -> SearchResult:
        logger.info(f"========== Starting Live Retrieval Pipeline for Search [{search_id}] ==========")

        try:
            # 1. Ensure the Embedder is ready in VRAM
            self._retrieval_phase.preload_model()

            # 2. Execute the Retrieval Phase
            queries: List[Tuple[uuid.UUID, SearchQuery]] = [(search_id, query)]
            phase_result = self._retrieval_phase.execute(queries)

            if phase_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: RAG Retrieval failed. Errors: {phase_result.errors}")

            # 3. Extract the transient SearchResult
            for job_result, search_result in phase_result.items:
                if isinstance(job_result, RagRetrievalJobResult) and job_result.success and search_result:
                    logger.info(f"========== Live Retrieval Pipeline Completed for Search [{search_id}] ==========")
                    return search_result

            raise WorkflowStateError(f"Pipeline halted: No search results returned from the database.")

        except WorkflowStateError as e:
            logger.error(f"Workflow State Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Critical Pipeline Crash: {e}")
            raise WorkflowExecutionError(f"Critical Pipeline Crash: {str(e)}", original_exception=e) from e
        finally:
            self._discard_pending_preloads()
