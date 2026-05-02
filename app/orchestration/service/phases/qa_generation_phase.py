# Canvas: qa_generation_phase_v1
# Path: app/orchestration/service/phases/qa_generation_phase.py

import logging
import concurrent.futures
import uuid
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

from app.features.qa_generation.service.generation_api import QAGenerationAPI
from app.features.qa_generation.domain.entities import QAResponse
from app.features.rag_retrieval.domain.entities import RetrievedChunk

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, QAGenerationJobResult

logger = logging.getLogger(__name__)


class QAGenerationPhase:
    def __init__(
            self,
            job_manager: JobManager,
            qa_api: QAGenerationAPI,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._qa_api = qa_api
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        logger.info("Triggering background preload for QA Generation LLM...")
        manifest = ModelManifest(
            model_type=ModelType.QA_GENERATION,
            model_path=settings.QA_GENERATION_MODEL_PATH,
            required_vram_mb=settings.QA_GENERATION_VRAM_REQUIREMENT_MB
        )
        self._preload_future = self._async_service.trigger_preload(
            manifest=manifest,
            loader=self._loader
        )

    def execute(
            self,
            chat_requests: List[Tuple[uuid.UUID, str, List[RetrievedChunk]]]
    ) -> PhaseBatchResult:
        """
        Executes the QA Generation process.
        Takes a list of tuples containing: (search_id, user_query, scrubbed_retrieved_chunks).
        """
        logger.info(f"Executing QA Generation Phase for {len(chat_requests)} queries...")

        # Output format explicitly pairs the Job tracking state with the pure domain Payload
        results: List[Tuple[QAGenerationJobResult, Optional[QAResponse]]] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            try:
                loaded_model = self._async_service.resolve_preload(self._preload_future)
                logger.info("Successfully resolved preloaded QA Generation LLM.")
            except ModelManagerError as e:
                logger.error(f"Failed to load QA Generation LLM: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.QA_GENERATION,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading QA Generation LLM: {str(e)}"]
                )

        try:
            for search_id, user_query, chunks in chat_requests:
                # Fallback to string literal if JobType enum hasn't been recompiled
                job_type = getattr(JobType, "QA_GENERATION", JobType("qa_generation"))

                request = JobRequest(
                    job_type=job_type,
                    source_id=None,
                    input_params={"search_id": str(search_id)}
                )

                try:
                    logger.info(f"Processing QA Generation for Search ID: {search_id}")

                    # We pass release_after=False so the batch loop handles VRAM retention
                    qa_response: QAResponse = self._job_manager.run_job(
                        request,
                        self._qa_api.generate,
                        user_query=user_query,
                        retrieved_chunks=chunks,
                        release_after=False
                    )

                    preview = qa_response.generated_answer[:50] + "..." if qa_response.generated_answer else ""

                    job_result = QAGenerationJobResult(
                        search_id=search_id,
                        success=True,
                        generated_answer_preview=preview,
                        citations_count=len(qa_response.citations)
                    )

                    results.append((job_result, qa_response))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Search ID {search_id}: {err_msg}")

                    job_result = QAGenerationJobResult(search_id=search_id, success=False, error_message=err_msg)
                    results.append((job_result, None))
                    errors.append(f"Failed QA Generation for Search ID {search_id}: {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                logger.info("Unloading QA Generation model to free VRAM...")
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
            phase=PipelinePhase.QA_GENERATION,
            status=status,
            total_processed=len(chat_requests),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )