# Path: app/orchestration/service/phases/document_contextualization_phase.py

import concurrent.futures
import logging
from typing import List, Optional

from app.core.config import settings
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType
from app.core.jobs.service.manager import JobManager
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.contextual_retrieval.service.contextualizer_api import ContextualizerAPI
from app.features.document_conversion.domain.entities import ConvertedDocument
from app.features.document_conversion.domain.interfaces import IConvertedDocumentRepository
from app.features.document_structuring.service.structuring_api import DocumentStructuringAPI
from app.orchestration.domain.entities import (
    ContextualRetrievalJobResult, DocumentConversionJobResult, PhaseBatchResult,
)
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase

logger = logging.getLogger(__name__)


class DocumentContextualizationPhase:
    """
    Cuts each converted document into section-aligned chunks and situates every chunk with the contextual SLM.

    Structuring is pure and takes milliseconds, so it runs here, inside the same job, rather than as a phase of
    its own with a table to hand its output over in. The SLM is loaded once for the whole batch, the same
    preload/resolve/unload shape as ContextualRetrievalPhase; the results are ContextualRetrievalJobResults, so
    ContextUnificationPhase and RagIndexingPhase take them unchanged.
    """

    def __init__(
            self,
            job_manager: JobManager,
            structuring_api: DocumentStructuringAPI,
            contextualizer_api: ContextualizerAPI,
            conversion_repo: IConvertedDocumentRepository,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._structuring_api = structuring_api
        self._contextualizer_api = contextualizer_api
        self._conversion_repo = conversion_repo
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        logger.info("Triggering background preload for the document contextualization SLM...")
        manifest = ModelManifest(
            model_type=ModelType.CONTEXTUAL_RETRIEVAL,
            model_path=settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH,
            required_vram_mb=settings.CONTEXTUAL_RETRIEVAL_VRAM_REQUIREMENT_MB
        )
        self._preload_future = self._async_service.trigger_preload(manifest=manifest, loader=self._loader)

    def discard_preload(self) -> None:
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._loader)

    def execute(self, conversion_results: List[DocumentConversionJobResult]) -> PhaseBatchResult:
        results: List[ContextualRetrievalJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded document contextualization SLM.")
            except ModelManagerError as e:
                logger.error(f"Failed to load the contextual SLM: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.DOCUMENT_CONTEXTUALIZATION,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading SLM model: {str(e)}"]
                )

        try:
            for item in conversion_results:
                if not item.success:
                    continue

                source_id = item.source_id
                document = self._conversion_repo.get_by_source_id(source_id)
                if not document:
                    err_msg = f"Cannot contextualize: converted document for Source {source_id} missing."
                    logger.error(err_msg)
                    results.append(ContextualRetrievalJobResult(source_id=source_id, success=False,
                                                                error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.DOCUMENT_CONTEXTUALIZATION,
                    source_id=source_id,
                    input_params={"document_source_id": str(source_id)}
                )
                try:
                    logger.info(f"Processing Document Contextualization for Source: {source_id}")
                    contextualized = self._job_manager.run_job(request, self._structure_and_contextualize,
                                                               document=document)
                    results.append(ContextualRetrievalJobResult(
                        source_id=source_id, success=True, contextual_transcript_id=contextualized.id
                    ))
                    success_count += 1
                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Source {source_id}: {err_msg}")
                    results.append(ContextualRetrievalJobResult(source_id=source_id, success=False,
                                                                error_message=err_msg))
                    errors.append(f"Failed Document Contextualization for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                logger.info("Unloading the contextual SLM to free VRAM...")
                self._async_service.trigger_unload(loaded_model=loaded_model, loader=self._loader, flush=True)

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.DOCUMENT_CONTEXTUALIZATION,
            status=status,
            total_processed=len(conversion_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )

    def _structure_and_contextualize(self, document: ConvertedDocument) -> ContextualizedTranscript:
        structured = self._structuring_api.execute(document)
        return self._contextualizer_api.execute_document(structured, release_after=False)
