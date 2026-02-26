# Canvas: diarize_phase_v1
# Path: app/orchestration/service/phases/diarize_phase.py

import concurrent.futures
from typing import List, Optional

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.diarization.service.diarization_api import DiarizationAPI
from app.features.storage.service.storage_api import StorageAPI

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, TranscribeJobResult, DiarizeJobResult


class DiarizePhase:
    def __init__(
            self,
            job_manager: JobManager,
            diarization_api: DiarizationAPI,
            storage_api: StorageAPI,
            async_model_service: AsyncModelService,
            diarize_manifest: ModelManifest,
            diarize_adapter: IModelLoader
    ):
        self._job_manager = job_manager
        self._diarization_api = diarization_api
        self._storage = storage_api
        self._async_service = async_model_service
        self._manifest = diarize_manifest
        self._adapter = diarize_adapter

        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        """Triggers the background loading of the Diarization model."""
        self._preload_future = self._async_service.trigger_preload(self._manifest, self._adapter)

    def execute(self, transcribe_results: List[TranscribeJobResult]) -> PhaseBatchResult:
        """
        Phase 6: Waits for the NeMo Diarization model to load, then diarizes sequentially.
        """
        results: List[DiarizeJobResult] = []
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
                    phase=PipelinePhase.DIARIZE,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Diarization model: {str(e)}"]
                )

        try:
            for item in transcribe_results:
                if not item.success or not item.transcript_id:
                    continue

                source_id = item.source_id

                # Fetch the original Source to find its linked audio file
                source = self._storage.get_source(source_id)
                if not source or not source.file:
                    err_msg = f"Cannot diarize: Audio file for Source {source_id} missing."
                    results.append(DiarizeJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.DIARIZE,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    # Execute API. `release_after=False` retains TitaNet in VRAM for the whole batch
                    diarization_map = self._job_manager.run_job(
                        request,
                        self._diarization_api.process_audio,
                        source_id=source_id,
                        audio_path=source.file.file_path,
                        release_after=False
                    )

                    results.append(DiarizeJobResult(source_id=source_id, success=True, diarization_map_id=diarization_map.id))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    results.append(DiarizeJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Diarization for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            # 2. Safely unload the Diarization model in the background once the entire batch is done
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
            phase=PipelinePhase.DIARIZE,
            status=status,
            total_processed=len(transcribe_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )