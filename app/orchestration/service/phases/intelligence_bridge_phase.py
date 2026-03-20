# Canvas: intelligence_bridge_phase_v1
# Path: app/orchestration/service/phases/intelligence_bridge_phase.py

import concurrent.futures
from typing import List, Optional

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.legacy_features.intelligence_bridge.service.bridge_api import IntelligenceBridgeAPI
from app.features.storage.service.storage_api import StorageAPI

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, DiarizeJobResult, IntelligenceBridgeJobResult


class IntelligenceBridgePhase:
    def __init__(
            self,
            job_manager: JobManager,
            bridge_api: IntelligenceBridgeAPI,
            storage_api: StorageAPI,
            async_model_service: AsyncModelService,
            nlp_manifest: ModelManifest,
            nlp_adapter: IModelLoader
    ):
        self._job_manager = job_manager
        self._bridge_api = bridge_api
        self._storage = storage_api
        self._async_service = async_model_service
        self._manifest = nlp_manifest
        self._adapter = nlp_adapter

        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        """Triggers the background loading of the Phi-3.5 NLP model."""
        self._preload_future = self._async_service.trigger_preload(self._manifest, self._adapter)

    def execute(self, diarize_results: List[DiarizeJobResult]) -> PhaseBatchResult:
        """
        Phase 7: Waits for the NLP model to load, then maps multimodal scenes sequentially.
        """
        results: List[IntelligenceBridgeJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        # 1. Wait for the background GPU loading to finish
        if self._preload_future:
            try:
                loaded_model = self._async_service.resolve_preload(self._preload_future)
            except ModelManagerError as e:
                return PhaseBatchResult(
                    phase=PipelinePhase.INTELLIGENCE_BRIDGE,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading NLP model: {str(e)}"]
                )

        try:
            for item in diarize_results:
                if not item.success:
                    continue

                source_id = item.source_id

                # Fetch the original Source to find its linked video file
                source = self._storage.get_source(source_id)
                if not source or not source.file:
                    err_msg = f"Cannot bridge intelligence: Video file for Source {source_id} missing."
                    results.append(IntelligenceBridgeJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.INTELLIGENCE_BRIDGE,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    # Execute API. The NLPRouter internal to the Bridge handles its own inference lifecycle.
                    scene_map = self._job_manager.run_job(
                        request,
                        self._bridge_api.process_multimodal_timeline,
                        source_id=source_id,
                        video_path=source.file.file_path
                    )

                    results.append(IntelligenceBridgeJobResult(source_id=source_id, success=True, scene_map_id=scene_map.id))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    results.append(IntelligenceBridgeJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Intelligence Bridge for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            # 2. Safely unload the NLP model in the background once the entire batch is done
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
            phase=PipelinePhase.INTELLIGENCE_BRIDGE,
            status=status,
            total_processed=len(diarize_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )