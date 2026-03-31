# Canvas: visual_cue_detection_phase_v1
# Path: app/orchestration/service/phases/visual_cue_detection_phase.py

import concurrent.futures
from typing import List, Optional

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.visual_cue_detection.service.detection_api import VisualCueDetectionAPI
from app.features.contextual_retrieval.domain.interfaces import IContextualRepository

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, ContextualRetrievalJobResult, VisualCueDetectionJobResult


class VisualCueDetectionPhase:
    def __init__(
            self,
            job_manager: JobManager,
            detection_api: VisualCueDetectionAPI,
            contextual_repo: IContextualRepository,
            async_model_service: AsyncModelService,
            manifest: ModelManifest,
            adapter: IModelLoader
    ):
        self._job_manager = job_manager
        self._detection_api = detection_api
        self._contextual_repo = contextual_repo
        self._async_service = async_model_service
        self._manifest = manifest
        self._adapter = adapter
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        """Triggers the background loading of the NLP gatekeeper model."""
        self._preload_future = self._async_service.trigger_preload(self._manifest, self._adapter)

    def execute(self, contextual_results: List[ContextualRetrievalJobResult]) -> PhaseBatchResult:
        """
        Phase 10: Waits for the SLM model to load, then evaluates all chunks across the entire batch
        for visual cues before unloading.
        """
        results: List[VisualCueDetectionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            try:
                loaded_model = self._async_service.resolve_preload(self._preload_future)
            except ModelManagerError as e:
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
                    results.append(VisualCueDetectionJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.VISUAL_CUE_DETECTION,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    # Execute API.
                    # The LoadedModel is passed directly so inference is fully contained here without model reload overhead.
                    evaluated_transcript = self._job_manager.run_job(
                        request,
                        self._detection_api.process_transcript,
                        transcript=transcript,
                        model=loaded_model
                    )

                    results.append(VisualCueDetectionJobResult(
                        source_id=source_id,
                        success=True,
                        evaluated_transcript_id=evaluated_transcript.id
                    ))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    results.append(VisualCueDetectionJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Visual Cue Detection for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            # Safely unload the NLP model in the background once the ENTIRE batch is done
            if loaded_model:
                self._async_service.trigger_unload(loaded_model, self._adapter, flush=True)

        if failed_count == 0 and success_count > 0:
            status = PhaseStatus.COMPLETED
        elif success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS
        else:
            status = PhaseStatus.COMPLETED

        return PhaseBatchResult(
            phase=PipelinePhase.VISUAL_CUE_DETECTION,
            status=status,
            total_processed=len(contextual_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )