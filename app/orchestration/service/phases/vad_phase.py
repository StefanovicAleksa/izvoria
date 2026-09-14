# filename: vad_phase_v1.py
# path: app/orchestration/service/phases/vad_phase.py
import concurrent.futures
from typing import List, Optional

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.vad.service.vad_api import VadAPI
from app.features.storage.service.storage_api import StorageAPI

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, AudioExtractionJobResult, VadJobResult


class VadPhase:
    def __init__(
            self,
            job_manager: JobManager,
            vad_api: VadAPI,
            storage_api: StorageAPI,
            async_model_service: AsyncModelService,
            vad_manifest: ModelManifest,
            vad_adapter: IModelLoader
    ):
        self._job_manager = job_manager
        self._vad_api = vad_api
        self._storage = storage_api
        self._async_service = async_model_service
        self._manifest = vad_manifest
        self._adapter = vad_adapter

        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        """Triggers the background loading of the VAD model."""
        self._preload_future = self._async_service.trigger_preload(self._manifest, self._adapter)

    def discard_preload(self) -> None:
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._adapter)

    def execute(self, extraction_results: List[AudioExtractionJobResult]) -> PhaseBatchResult:
        """
        Phase 4: Waits for the background model to load, then processes VAD sequentially.
        """
        results: List[VadJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        # 1. Wait for the background GPU loading to finish
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                # This will instantly return if the load finished during Phase 3
                loaded_model = self._async_service.resolve_preload(future)
            except ModelManagerError as e:
                # Catch specific domain errors like OutOfVramError or ModelLoadError
                return PhaseBatchResult(
                    phase=PipelinePhase.VAD,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading VAD model: {str(e)}"]
                )

        try:
            for item in extraction_results:
                if not item.success or not item.extracted_file_id:
                    continue

                source_id = item.source_id

                # Retrieve physical file path
                audio_file = self._storage.get_file(item.extracted_file_id)
                if not audio_file:
                    err_msg = f"Audio file ID {item.extracted_file_id} not found."
                    results.append(VadJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.VAD,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id), "audio_path": audio_file.file_path}
                )

                try:
                    # Execute API. Notice `release_after=False` so the batch retains the VRAM map
                    vad_map = self._job_manager.run_job(
                        request,
                        self._vad_api.process_audio,
                        source_id=source_id,
                        audio_path=audio_file.file_path,
                        release_after=False
                    )

                    results.append(VadJobResult(source_id=source_id, success=True, vad_map_id=vad_map.id))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    results.append(VadJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed VAD mapping for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            # 2. Safely unload the model in the background once the entire batch is done
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
            phase=PipelinePhase.VAD,
            status=status,
            total_processed=len(extraction_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )