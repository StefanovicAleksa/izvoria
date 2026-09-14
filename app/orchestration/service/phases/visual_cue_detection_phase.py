# Canvas: visual_cue_detection_phase_v7
# Path: app/orchestration/service/phases/visual_cue_detection_phase.py

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

from app.features.visual_cue_detection.service.detection_api import VisualCueDetectionAPI
from app.features.contextual_retrieval.domain.interfaces import IContextualRepository
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, ContextualRetrievalJobResult, \
    VisualCueDetectionJobResult

logger = logging.getLogger(__name__)


class VisualCueDetectionPhase:
    def __init__(
            self,
            job_manager: JobManager,
            detection_api: VisualCueDetectionAPI,
            contextual_repo: IContextualRepository,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._detection_api = detection_api
        self._contextual_repo = contextual_repo
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        logger.info("Triggering background preload for Visual Cue Detection model...")

        manifest = ModelManifest(
            model_type=ModelType.VISUAL_CUE_DETECTION,
            model_path=settings.VISUAL_CUE_DETECTION_MODEL_PATH,
            required_vram_mb=settings.VISUAL_CUE_DETECTION_VRAM_REQUIREMENT_MB
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

    def execute(self, contextual_results: List[ContextualRetrievalJobResult]) -> PhaseBatchResult:
        logger.info(f"Executing Visual Cue Detection Phase for {len(contextual_results)} transcripts...")
        results: List[VisualCueDetectionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded Visual Cue Detection model.")
            except ModelManagerError as e:
                logger.error(f"Failed to load Visual Cue SLM: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.VISUAL_CUE_DETECTION,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Visual Cue SLM: {str(e)}"]
                )

        try:
            for item in contextual_results:
                if not item.success:
                    continue

                source_id = item.source_id
                transcript = self._contextual_repo.get_by_source_id(source_id)

                if not transcript:
                    err_msg = f"Cannot detect visual cues: Contextual Transcript for Source {source_id} missing."
                    logger.error(err_msg)
                    results.append(
                        VisualCueDetectionJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.VISUAL_CUE_DETECTION,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    logger.info(f"Processing Visual Cue Detection for Source: {source_id}")
                    evaluated_transcript = self._job_manager.run_job(
                        request,
                        self._detection_api.process_transcript,
                        transcript=transcript,
                        model=loaded_model,
                        release_after=False  # CRITICAL FIX: Tell the API to leave the model in VRAM for the next batch item
                    )

                    results.append(VisualCueDetectionJobResult(
                        source_id=source_id,
                        success=True,
                        evaluated_transcript_id=evaluated_transcript.id
                    ))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Source {source_id}: {err_msg}")
                    results.append(
                        VisualCueDetectionJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Visual Cue Detection for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            # Safely unload the NLP model in the background once the ENTIRE batch is done
            if loaded_model:
                logger.info("Unloading Visual Cue Detection model to free VRAM...")
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
            phase=PipelinePhase.VISUAL_CUE_DETECTION,
            status=status,
            total_processed=len(contextual_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )