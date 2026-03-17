# Canvas: transcribe_phase_v3
# Path: app/orchestration/service/phases/transcribe_phase.py

import concurrent.futures
from typing import List, Optional

from app.core.config import settings
from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.storage.service.storage_api import StorageAPI
from app.features.domain_adaptation.domain.interfaces import IDomainAdaptationRepository

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, VadJobResult, TranscribeJobResult


class TranscribePhase:
    def __init__(
            self,
            job_manager: JobManager,
            transcription_api: TranscriptionAPI,
            storage_api: StorageAPI,
            domain_repo: IDomainAdaptationRepository,
            async_model_service: AsyncModelService,
            transcribe_manifest: ModelManifest,
            transcribe_adapter: IModelLoader
    ):
        self._job_manager = job_manager
        self._transcription_api = transcription_api
        self._storage = storage_api
        self._domain_repo = domain_repo
        self._async_service = async_model_service
        self._manifest = transcribe_manifest
        self._adapter = transcribe_adapter

        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        """Triggers the background loading of the Transcription model."""
        self._preload_future = self._async_service.trigger_preload(self._manifest, self._adapter)

    def execute(self, vad_results: List[VadJobResult]) -> PhaseBatchResult:
        """
        Phase 5: Waits for the Whisper model to load, then transcribes sequentially with Domain Hints.
        """
        results: List[TranscribeJobResult] = []
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
                    phase=PipelinePhase.TRANSCRIBE,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Transcription model: {str(e)}"]
                )

        try:
            for item in vad_results:
                if not item.success or not item.vad_map_id:
                    continue

                source_id = item.source_id

                source = self._storage.get_source(source_id)
                if not source or not source.file:
                    err_msg = f"Cannot transcribe: Audio file for Source {source_id} missing."
                    results.append(TranscribeJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                # 2. Fetch Domain Adaptation Hints (if they exist)
                glossary = self._domain_repo.get_glossary_by_source_id(source_id)
                hotwords = glossary.hotword_string if glossary else None
                language = settings.APPLIANCE_LANGUAGE

                request = JobRequest(
                    job_type=JobType.TRANSCRIBE,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    # Execute API. `release_after=False` retains Whisper in VRAM for the whole batch
                    transcript = self._job_manager.run_job(
                        request,
                        self._transcription_api.process_audio,
                        source_id=source_id,
                        audio_path=source.file.file_path,
                        language=language,
                        hotwords=hotwords,
                        release_after=False
                    )

                    results.append(TranscribeJobResult(source_id=source_id, success=True, transcript_id=transcript.id))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    results.append(TranscribeJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Transcription for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            # 3. Safely unload the Whisper model in the background once the entire batch is done
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
            phase=PipelinePhase.TRANSCRIBE,
            status=status,
            total_processed=len(vad_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )