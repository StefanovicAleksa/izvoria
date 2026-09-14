# Canvas: rag_indexing_phase_v2
# Path: app/orchestration/service/phases/rag_indexing_phase.py

import logging
import concurrent.futures
from typing import List, Optional

from app.core.config import settings
from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.rag_storage.service.storage_api import RAGStorageService
from app.features.context_unification.domain.interfaces import IContextUnificationRepository

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, ContextUnificationJobResult, RagIndexingJobResult

logger = logging.getLogger(__name__)


class RagIndexingPhase:
    def __init__(
            self,
            job_manager: JobManager,
            rag_storage_service: RAGStorageService,
            unified_repo: IContextUnificationRepository,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._rag_storage_service = rag_storage_service
        self._unified_repo = unified_repo
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        logger.info("Triggering background preload for RAG Embedding model...")
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

    def execute(self, unification_results: List[ContextUnificationJobResult]) -> PhaseBatchResult:
        logger.info(f"Executing RAG Indexing Phase for {len(unification_results)} transcripts...")
        results: List[RagIndexingJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded RAG Embedding model.")
            except ModelManagerError as e:
                logger.error(f"Failed to load RAG Embedding model: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.RAG_INDEXING,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading RAG Embedding model: {str(e)}"]
                )

        try:
            for item in unification_results:
                if not item.success:
                    continue

                source_id = item.source_id

                transcript = self._unified_repo.get_by_source_id(source_id)
                if not transcript:
                    err_msg = f"Cannot index vectors: Unified Transcript for Source {source_id} missing."
                    logger.error(err_msg)
                    results.append(RagIndexingJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.RAG_INDEXING,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    logger.info(f"Processing RAG Indexing (Hybrid Vector + FTS) for Source: {source_id}")

                    # release_after=False keeps it in VRAM for the batch loop. We unload in finally block.
                    ingestion_result = self._job_manager.run_job(
                        request,
                        self._rag_storage_service.index_transcript,
                        source_id=source_id,
                        transcript=transcript,
                        release_after=False
                    )

                    if ingestion_result.success:
                        results.append(RagIndexingJobResult(
                            source_id=source_id,
                            success=True,
                            dense_vectors_saved=ingestion_result.telemetry.dense_vectors_saved,
                            sparse_vectors_saved=ingestion_result.telemetry.sparse_vectors_saved
                        ))
                        success_count += 1
                    else:
                        err_msg = ingestion_result.error_message or "Unknown ingestion failure."
                        results.append(RagIndexingJobResult(source_id=source_id, success=False, error_message=err_msg))
                        errors.append(f"Failed RAG Indexing for Source {source_id}: {err_msg}")
                        failed_count += 1

                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Source {source_id}: {err_msg}")
                    results.append(RagIndexingJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed RAG Indexing for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                logger.info("Unloading RAG Embedding model to flush VRAM for LLM inference...")
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
            phase=PipelinePhase.RAG_INDEXING,
            status=status,
            total_processed=len(unification_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )