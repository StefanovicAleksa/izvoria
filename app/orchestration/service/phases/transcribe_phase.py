# Canvas: transcribe_phase_v5
# Path: app/orchestration/service/phases/transcribe_phase.py

import concurrent.futures
import logging
from typing import List, Optional

from app.core.language.service.language_api import get_language_pack
from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.storage.service.storage_api import StorageAPI
from app.features.transcription.exceptions import LanguageMismatchError
from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI
from app.features.domain_adaptation.domain.interfaces import IDomainAdaptationRepository

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, VadJobResult, TranscribeJobResult

logger = logging.getLogger(__name__)


class TranscribePhase:
    def __init__(
            self,
            job_manager: JobManager,
            transcription_api: TranscriptionAPI,
            audio_extractor: AudioExtractorAPI,
            domain_repo: IDomainAdaptationRepository,
            async_model_service: AsyncModelService,
            transcribe_manifest: ModelManifest,
            transcribe_adapter: IModelLoader,
            storage_api: Optional[StorageAPI] = None
    ):
        self._job_manager = job_manager
        self._transcription_api = transcription_api
        self._audio_extractor = audio_extractor
        self._domain_repo = domain_repo
        self._async_service = async_model_service
        self._manifest = transcribe_manifest
        self._adapter = transcribe_adapter
        self._storage = storage_api

        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        self._preload_future = self._async_service.trigger_preload(self._manifest, self._adapter)

    def discard_preload(self) -> None:
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._adapter)

    def _forget_language(self, source_id) -> None:
        """
        A file whose speech is in another language than the batch's was stamped with the wrong one, and its
        glossary was built from a scout pass in that language. Both are forgotten, so ingesting the file again
        with the right language is accepted and starts from a clean glossary. VAD and diarization don't depend
        on the language and are kept.
        """
        try:
            self._domain_repo.delete_glossary_by_source_id(source_id)
            if self._storage:
                self._storage.clear_language(source_id)
        except Exception as ex:
            logger.warning(f"Could not un-stamp Source {source_id} after its language check failed: {ex}")

    def execute(self, vad_results: List[VadJobResult]) -> PhaseBatchResult:
        results: List[TranscribeJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
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

                try:
                    audio_path = self._audio_extractor.get_extracted_audio_path(source_id)
                except Exception as e:
                    err_msg = f"Cannot transcribe: extracted audio for Source {source_id} missing. {e}"
                    results.append(TranscribeJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                try:
                    glossary = self._domain_repo.get_glossary_by_source_id(source_id)
                    hotwords = getattr(glossary, "hotword_string", None) if glossary else None
                    initial_prompt = getattr(glossary, "initial_prompt_string", None) if glossary else None
                except Exception as ex:
                    logger.warning(f"Failed to fetch Domain Hints for {source_id}: {str(ex)}")
                    hotwords, initial_prompt = None, None

                language = get_language_pack().transcription.whisper_language

                request = JobRequest(
                    job_type=JobType.TRANSCRIBE,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    transcript = self._job_manager.run_job(
                        request,
                        self._transcription_api.process_audio,
                        source_id=source_id,
                        audio_path=audio_path,
                        language=language,
                        hotwords=hotwords,
                        initial_prompt=initial_prompt,
                        release_after=False
                    )

                    results.append(TranscribeJobResult(source_id=source_id, success=True, transcript_id=transcript.id))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    # Logged with the traceback attached: transcription failures are the ones most often
                    # diagnosed after the fact from a log file, where a bare message is not enough.
                    logger.error(
                        f"TranscribePhase failed processing audio for source {source_id}: {err_msg}",
                        exc_info=True,
                    )
                    results.append(TranscribeJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Transcription for Source {source_id}: {err_msg}")
                    failed_count += 1
                    if isinstance(e, LanguageMismatchError):
                        self._forget_language(source_id)

        finally:
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