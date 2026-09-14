# Canvas: retrieval_scrubber_phase_v2
# Path: app/orchestration/service/phases/retrieval_scrubber_phase.py

import logging
import concurrent.futures
from typing import List, Optional, Tuple

from app.core.config import settings
from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.rag_retrieval.domain.entities import SearchResult, RetrievedChunk
from app.features.retrieval_hallucination_scrubber.service.scrubber_api import RetrievalScrubberAPI

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, RagRetrievalJobResult, RetrievalScrubbingJobResult

logger = logging.getLogger(__name__)


class RetrievalScrubberPhase:
    def __init__(
            self,
            job_manager: JobManager,
            scrubber_api: RetrievalScrubberAPI,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._scrubber_api = scrubber_api
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        logger.info("Triggering background preload for Retrieval Scrubber SLM...")
        manifest = ModelManifest(
            model_type=ModelType.RETRIEVAL_SCRUBBER,
            model_path=settings.RETRIEVAL_SCRUBBER_MODEL_PATH,
            required_vram_mb=settings.RETRIEVAL_SCRUBBER_VRAM_REQUIREMENT_MB
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

    def execute(
            self,
            retrieval_results: List[Tuple[RagRetrievalJobResult, Optional[SearchResult]]]
    ) -> PhaseBatchResult:
        logger.info(f"Executing Retrieval Scrubber Phase for {len(retrieval_results)} queries...")

        # We output tuples of (JobResult, Surviving Chunks) to pass to the Final LLM Synthesis
        results: List[Tuple[RetrievalScrubbingJobResult, List[RetrievedChunk]]] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded Retrieval Scrubber SLM.")
            except ModelManagerError as e:
                logger.error(f"Failed to load Retrieval Scrubber SLM: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.RETRIEVAL_SCRUBBING,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Scrubber SLM: {str(e)}"]
                )

        try:
            for job_result, search_result in retrieval_results:
                if not job_result.success or not search_result:
                    continue

                search_id = job_result.search_id

                # We use a string fallback "retrieval_scrubbing"
                job_type = getattr(JobType, "RETRIEVAL_SCRUBBING", JobType("retrieval_scrubbing"))

                request = JobRequest(
                    job_type=job_type,
                    source_id=None,
                    input_params={"search_id": str(search_id)}
                )

                try:
                    logger.info(f"Processing Retrieval Scrubbing for Search ID: {search_id}")

                    # NEW: Accept explicit arguments to solve nested closure type-narrowing issues
                    def _scrub_wrapper(prompt: str, chunks: List[RetrievedChunk]) -> dict:
                        surviving, audits = self._scrubber_api.scrub_retrieved_chunks(
                            user_prompt=prompt,
                            raw_chunks=chunks,
                            release_after=False  # Handled by finally block
                        )
                        return {"surviving_chunks_count": len(surviving), "_transient_chunks": surviving}

                    # NEW: Pass the variables from the outer scope via kwargs
                    output_data = self._job_manager.run_job(
                        request,
                        _scrub_wrapper,
                        prompt=search_result.query,
                        chunks=search_result.results
                    )

                    surviving_chunks = output_data.get("_transient_chunks", [])

                    scrub_job_result = RetrievalScrubbingJobResult(
                        search_id=search_id,
                        success=True,
                        surviving_chunks_count=len(surviving_chunks)
                    )

                    results.append((scrub_job_result, surviving_chunks))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Search ID {search_id}: {err_msg}")

                    scrub_job_result = RetrievalScrubbingJobResult(search_id=search_id, success=False,
                                                                   error_message=err_msg)
                    results.append((scrub_job_result, []))
                    errors.append(f"Failed Retrieval Scrubbing for Search ID {search_id}: {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                logger.info("Unloading Retrieval Scrubber model to free VRAM for LLM Synthesis...")
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
            phase=PipelinePhase.RETRIEVAL_SCRUBBING,
            status=status,
            total_processed=len(retrieval_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )