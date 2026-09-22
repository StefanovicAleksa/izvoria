# Path: app/orchestration/service/phases/rag_reranking_phase.py

import logging
import concurrent.futures
from typing import Any, List, Optional, Tuple

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.rag_reranking.service.rerank_api import RerankAPI
from app.features.rag_retrieval.domain.entities import SearchResult

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, RagRerankingJobResult

logger = logging.getLogger(__name__)


class RagRerankingPhase:
    def __init__(
            self,
            job_manager: JobManager,
            rerank_api: RerankAPI,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._rerank_api = rerank_api
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        # Discard any preload a previous call fired but nobody resolved. Without this, overwriting
        # `_preload_future` below makes the old one unreachable: the background worker still loads that
        # model, `discard_preload` can never be called on it, nothing releases it, and the next
        # `acquire_model(..., wait_for_clear=True)` parks forever with no exception and no way out but a
        # restart. Straight-line pipelines resolve every preload exactly once so this never fired; the
        # report mode's gather loop calls preload_model once per round, which makes it load-bearing.
        self.discard_preload()
        if not self._rerank_api.is_available():
            logger.info("Reranker weights not installed; skipping preload (results keep RRF order).")
            return
        logger.info("Triggering background preload for RAG Cross-Encoder Reranker...")
        self._preload_future = self._async_service.trigger_preload(
            manifest=self._rerank_api.manifest,
            loader=self._loader
        )

    def discard_preload(self) -> None:
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._loader)

    def execute(
            self,
            retrieval_results: List[Tuple[Any, Optional[SearchResult]]],
            top_n: int
    ) -> PhaseBatchResult:
        """
        Reranks each successful retrieval's candidate pool down to `top_n` chunks.
        Emits (RagRerankingJobResult, reranked SearchResult) tuples for the scrubber / final output.
        """
        logger.info(f"Executing RAG Reranking Phase for {len(retrieval_results)} queries...")

        results: List[Tuple[RagRerankingJobResult, Optional[SearchResult]]] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded Reranker model.")
            except ModelManagerError as e:
                logger.error(f"Failed to load Reranker model: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.RAG_RERANKING,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Reranker model: {str(e)}"]
                )

        try:
            for job_result, search_result in retrieval_results:
                if not job_result.success or not search_result:
                    continue

                search_id = job_result.search_id
                request = JobRequest(
                    job_type=JobType.RAG_RERANKING,
                    source_id=None,
                    input_params={"search_id": str(search_id), "candidates": len(search_result.results)}
                )

                try:
                    reranked_chunks = self._job_manager.run_job(
                        request,
                        self._rerank_api.rerank,
                        query=search_result.query,
                        chunks=search_result.results,
                        top_n=top_n,
                        # Keep the model for the batch only when this phase owns a resolved preload
                        release_after=loaded_model is None
                    )

                    reranked_result = search_result.model_copy(update={"results": reranked_chunks})
                    results.append((
                        RagRerankingJobResult(
                            search_id=search_id,
                            success=True,
                            reranked_chunks_count=len(reranked_chunks)
                        ),
                        reranked_result
                    ))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Search ID {search_id}: {err_msg}")
                    results.append((RagRerankingJobResult(search_id=search_id, success=False, error_message=err_msg), None))
                    errors.append(f"Failed RAG Reranking for Search ID {search_id}: {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                logger.info("Unloading Reranker model to free VRAM for the next stage...")
                self._async_service.trigger_unload(
                    loaded_model=loaded_model,
                    loader=self._loader,
                    flush=True
                )

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.RAG_RERANKING,
            status=status,
            total_processed=len(retrieval_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )
