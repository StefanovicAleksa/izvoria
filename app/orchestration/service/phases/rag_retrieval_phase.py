# Canvas: rag_retrieval_phase_v1
# Path: app/orchestration/service/phases/rag_retrieval_phase.py

import logging
import concurrent.futures
import uuid
from typing import List, Optional, Tuple

from app.core.config import settings
from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import CancellationToken, JobRequest
from app.core.jobs.domain.enums import JobType
from app.core.jobs.exceptions.execution import JobCancelled

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI
from app.features.rag_retrieval.domain.entities import (
    BatchSearchResult, CorpusSweep, IndexedSource, RetrievedChunk, SearchQuery, SearchResult,
)

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import (
    EvidenceSweepJobResult, PhaseBatchResult, RagRetrievalJobResult,
)

logger = logging.getLogger(__name__)


class RagRetrievalPhase:
    def __init__(
            self,
            job_manager: JobManager,
            retrieval_api: RetrievalAPI,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._retrieval_api = retrieval_api
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
        logger.info("Triggering background preload for RAG Prompt Embedder...")
        manifest = ModelManifest(
            model_type=ModelType.RAG_EMBEDDING,
            model_path=settings.RAG_EMBEDDING_MODEL_PATH,
            required_vram_mb=settings.RAG_EMBEDDING_VRAM_REQUIREMENT_MB
        )
        self._preload_future = self._async_service.trigger_preload(
            manifest=manifest,
            loader=self._loader
        )

    def discard_preload(self) -> None:
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._loader)

    def expand_neighbors(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Adds the chunks around the best `RAG_NEIGHBOR_EXPANSION_TOP_K` chunks. Extra context is optional, so a
        failed lookup keeps the chunks as they are instead of failing the answer.
        """
        try:
            return self._retrieval_api.expand_neighbors(chunks, settings.RAG_NEIGHBOR_EXPANSION_TOP_K)
        except Exception as e:
            logger.error(f"Neighbour expansion failed; continuing with the retrieved chunks only: {e}")
            return list(chunks)

    def get_chunks(self, chunk_ids: List[uuid.UUID]) -> List[RetrievedChunk]:
        """
        Transcripts behind a set of chunk ids. Needs no model, so it is not a job -- it is how a resumed
        run rehydrates a persisted evidence bank without touching the embedder.
        """
        return self._retrieval_api.get_chunks(chunk_ids)

    def list_indexed_sources(self) -> List[IndexedSource]:
        """
        What the corpus actually holds. Needs no model, so it is not a job -- it is a read the scoping
        stage does before deciding anything.
        """
        return self._retrieval_api.list_indexed_sources()

    def execute_sweep(
            self,
            report_id: uuid.UUID,
            queries: List[Tuple[str, SearchQuery]],
            sweep: Optional[CorpusSweep] = None,
            candidates: Optional[int] = None,
            cancel_token: Optional[CancellationToken] = None,
    ) -> Tuple[PhaseBatchResult, BatchSearchResult]:
        """
        The report mode's whole gathering stage, under ONE embedder residency.

        Every sub-question AND the per-document sweep are embedded together and searched together. Running
        them as separate `execute()` calls would be the obvious thing and would cost one 2.2 GB load each;
        here the cost is one load no matter how wide the fan-out gets.

        `queries` pairs each sub-question's text with the query to run, so a hit can be attributed back
        to the question that found it in the evidence bank.

        Returns the phase result alongside the raw BatchSearchResult, because the caller needs the actual
        chunks -- PhaseBatchResult.items only carries the per-job bookkeeping.
        """
        logger.info(
            f"Executing evidence sweep for report {report_id}: {len(queries)} sub-question(s)"
            f"{' plus a per-document sweep' if sweep else ''}."
        )

        results: List[Tuple[EvidenceSweepJobResult, Optional[SearchResult]]] = []
        errors: List[str] = []
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded Embedder model for the evidence sweep.")
            except ModelManagerError as e:
                logger.error(f"Failed to load Embedder model: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.RESEARCH_SWEEP,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Embedder model: {str(e)}"]
                ), BatchSearchResult()

        batch = BatchSearchResult()
        try:
            request = JobRequest(
                job_type=JobType.RAG_RETRIEVAL,
                source_id=None,
                input_params={"report_id": str(report_id), "queries": len(queries)},
                progress_detail=f"Searching the corpus ({len(queries)} queries)",
            )
            batch = self._job_manager.run_job(
                request,
                self._retrieval_api.search_batch,
                cancel_token=cancel_token,
                queries=[query for _, query in queries],
                sweep=sweep,
                candidates=candidates,
            )

            for (question_text, query), result in zip(queries, batch.results):
                results.append((
                    EvidenceSweepJobResult(
                        report_id=report_id,
                        success=True,
                        query=question_text,
                        chunks_found=len(result.results),
                    ),
                    result,
                ))

            if sweep is not None:
                results.append((
                    EvidenceSweepJobResult(
                        report_id=report_id,
                        success=True,
                        query=sweep.prompt,
                        chunks_found=len(batch.sweep_results),
                        is_sweep=True,
                        swept_source_count=batch.swept_source_count,
                    ),
                    None,
                ))

        except JobCancelled:
            # Cancellation is a decision, not a failure: it must reach the pipeline so the run stops and
            # the report is recorded as CANCELLED, rather than being flattened into a failed search here.
            raise

        except Exception as e:
            err_msg = str(e)
            logger.error(f"Evidence sweep failed for report {report_id}: {err_msg}")
            errors.append(f"Evidence sweep failed for report {report_id}: {err_msg}")
            results.append((
                EvidenceSweepJobResult(
                    report_id=report_id, success=False, query="", error_message=err_msg
                ),
                None,
            ))

        finally:
            if loaded_model:
                logger.info("Unloading Embedder model to free VRAM for the reranker...")
                self._async_service.trigger_unload(
                    loaded_model=loaded_model,
                    loader=self._loader,
                    flush=True
                )

        success_count = sum(1 for job_result, _ in results if job_result.success)
        failed_count = len(results) - success_count
        status = PhaseStatus.COMPLETED
        if success_count == 0:
            status = PhaseStatus.FAILED
        elif failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.RESEARCH_SWEEP,
            status=status,
            total_processed=len(results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors,
        ), batch

    def execute(self, queries: List[Tuple[uuid.UUID, SearchQuery]]) -> PhaseBatchResult:
        logger.info(f"Executing RAG Retrieval Phase for {len(queries)} queries...")

        # We store tuples of (JobResult, Actual SearchResult) to pass memory-bound data to the scrubber
        results: List[Tuple[RagRetrievalJobResult, Optional[SearchResult]]] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded Embedder model.")
            except ModelManagerError as e:
                logger.error(f"Failed to load Embedder model: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.RAG_RETRIEVAL,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Embedder model: {str(e)}"]
                )

        try:
            for search_id, query in queries:
                # We use a string fallback "rag_retrieval" in case the enum is missing in legacy environments
                job_type = getattr(JobType, "RAG_RETRIEVAL", JobType("rag_retrieval"))

                request = JobRequest(
                    job_type=job_type,
                    source_id=None,  # Live inference jobs aren't strictly bound to a single source_id
                    input_params={"search_id": str(search_id), "prompt": query.prompt}
                )

                try:
                    logger.info(f"Processing RAG Retrieval for Search ID: {search_id}")

                    # Execute the RAG Search
                    search_result: SearchResult = self._job_manager.run_job(
                        request,
                        self._retrieval_api.search,
                        query=query
                    )

                    job_result = RagRetrievalJobResult(
                        search_id=search_id,
                        success=True,
                        retrieved_chunks_count=len(search_result.results)
                    )

                    results.append((job_result, search_result))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Search ID {search_id}: {err_msg}")

                    job_result = RagRetrievalJobResult(search_id=search_id, success=False, error_message=err_msg)
                    results.append((job_result, None))
                    errors.append(f"Failed RAG Retrieval for Search ID {search_id}: {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                logger.info("Unloading Embedder model to free VRAM for SLM Scrubber...")
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
            phase=PipelinePhase.RAG_RETRIEVAL,
            status=status,
            total_processed=len(queries),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )