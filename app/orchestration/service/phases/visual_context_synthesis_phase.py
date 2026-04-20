# Canvas: visual_context_synthesis_phase_v1
# Path: app/orchestration/service/phases/visual_context_synthesis_phase.py

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

from app.features.visual_context_synthesis.service.synthesis_api import VisualSynthesisAPI
from app.features.contextual_retrieval.domain.interfaces import IContextualRepository
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, VisualHallucinationCorrectionJobResult, \
    VisualContextSynthesisJobResult

logger = logging.getLogger(__name__)


class VisualContextSynthesisPhase:
    def __init__(
            self,
            job_manager: JobManager,
            synthesis_api: VisualSynthesisAPI,
            contextual_repo: IContextualRepository,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._synthesis_api = synthesis_api
        self._contextual_repo = contextual_repo
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        logger.info("Triggering background preload for Visual Context Synthesis SLM...")
        manifest = ModelManifest(
            model_type=ModelType.VISUAL_CONTEXT_SYNTHESIS,
            model_path=settings.VISUAL_CONTEXT_SYNTHESIS_MODEL_PATH,
            required_vram_mb=settings.VISUAL_CONTEXT_SYNTHESIS_VRAM_REQUIREMENT_MB
        )
        self._preload_future = self._async_service.trigger_preload(
            manifest=manifest,
            loader=self._loader
        )

    def execute(self, hallucination_results: List[VisualHallucinationCorrectionJobResult]) -> PhaseBatchResult:
        logger.info(f"Executing Visual Context Synthesis Phase for {len(hallucination_results)} transcripts...")
        results: List[VisualContextSynthesisJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            try:
                loaded_model = self._async_service.resolve_preload(self._preload_future)
                logger.info("Successfully resolved preloaded Visual Context Synthesis SLM.")
            except ModelManagerError as e:
                logger.error(f"Failed to load Visual Synthesis SLM: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.VISUAL_CONTEXT_SYNTHESIS,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Visual Synthesis SLM: {str(e)}"]
                )

        try:
            for item in hallucination_results:
                if not item.success:
                    continue

                source_id = item.source_id

                transcript = self._contextual_repo.get_by_source_id(source_id)
                if not transcript:
                    err_msg = f"Cannot synthesize visual context: Contextual Transcript for Source {source_id} missing."
                    logger.error(err_msg)
                    results.append(
                        VisualContextSynthesisJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.VISUAL_CONTEXT_SYNTHESIS,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    logger.info(f"Processing Visual Context Synthesis for Source: {source_id}")
                    timeline = self._job_manager.run_job(
                        request,
                        self._synthesis_api.execute,
                        transcript=transcript,
                        release_after=False
                    )

                    results.append(VisualContextSynthesisJobResult(
                        source_id=source_id,
                        success=True,
                        visual_timeline_id=timeline.id
                    ))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Source {source_id}: {err_msg}")
                    results.append(
                        VisualContextSynthesisJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Visual Context Synthesis for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                logger.info("Unloading Visual Context Synthesis model to free VRAM...")
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
            phase=PipelinePhase.VISUAL_CONTEXT_SYNTHESIS,
            status=status,
            total_processed=len(hallucination_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )