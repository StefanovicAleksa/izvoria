# Canvas: visual_hallucination_correction_phase_v1
# Path: app/orchestration/service/phases/visual_hallucination_correction_phase.py

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

from app.features.visual_hallucination_correction.service.correction_api import VisualHallucinationCorrectionAPI
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, VisualAnalysisJobResult, \
    VisualHallucinationCorrectionJobResult

logger = logging.getLogger(__name__)


class VisualHallucinationCorrectionPhase:
    def __init__(
            self,
            job_manager: JobManager,
            correction_api: VisualHallucinationCorrectionAPI,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._correction_api = correction_api
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        logger.info("Triggering background preload for Visual Hallucination SLM Gatekeeper...")

        manifest = ModelManifest(
            model_type=ModelType.VISUAL_HALLUCINATION_CORRECTION,
            model_path=settings.VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH,
            required_vram_mb=settings.VISUAL_HALLUCINATION_CORRECTION_VRAM_REQUIREMENT_MB
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

    def execute(self, visual_analysis_results: List[VisualAnalysisJobResult]) -> PhaseBatchResult:
        logger.info(f"Executing Visual Hallucination Correction Phase for {len(visual_analysis_results)} transcripts...")
        results: List[VisualHallucinationCorrectionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded Visual Hallucination SLM.")
            except ModelManagerError as e:
                logger.error(f"Failed to load Visual Hallucination SLM: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.VISUAL_HALLUCINATION_CORRECTION,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Visual Hallucination SLM: {str(e)}"]
                )

        try:
            for item in visual_analysis_results:
                if not item.success:
                    continue

                source_id = item.source_id

                request = JobRequest(
                    job_type=JobType.VISUAL_HALLUCINATION_CORRECTION,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    logger.info(f"Processing Visual Hallucination Correction for Source: {source_id}")
                    evaluated_transcript = self._job_manager.run_job(
                        request,
                        self._correction_api.execute,
                        source_id=source_id,
                        release_after=False  # Keep model resident in VRAM for the entire batch loop
                    )

                    results.append(VisualHallucinationCorrectionJobResult(
                        source_id=source_id,
                        success=True,
                        evaluated_transcript_id=evaluated_transcript.id
                    ))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Source {source_id}: {err_msg}")
                    results.append(
                        VisualHallucinationCorrectionJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Visual Hallucination Correction for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                logger.info("Unloading Visual Hallucination model to free VRAM...")
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
            phase=PipelinePhase.VISUAL_HALLUCINATION_CORRECTION,
            status=status,
            total_processed=len(visual_analysis_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )