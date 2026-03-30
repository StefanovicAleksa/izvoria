# Canvas: contextual_retrieval_phase_v1
# Path: app/orchestration/service/phases/contextual_retrieval_phase.py

import concurrent.futures
from typing import List, Optional

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.contextual_retrieval.service.contextualizer_api import ContextualizerAPI
from app.features.phonetic_correction.domain.interfaces import IPhoneticRepository

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, PhoneticCorrectionJobResult, \
    ContextualRetrievalJobResult


class ContextualRetrievalPhase:
    def __init__(
            self,
            job_manager: JobManager,
            contextualizer_api: ContextualizerAPI,
            phonetic_repo: IPhoneticRepository,
            async_model_service: AsyncModelService,
            manifest: ModelManifest,
            adapter: IModelLoader
    ):
        self._job_manager = job_manager
        self._contextualizer_api = contextualizer_api
        self._phonetic_repo = phonetic_repo
        self._async_service = async_model_service
        self._manifest = manifest
        self._adapter = adapter
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        """Triggers the background loading of the SLM (Llama 3.2)."""
        self._preload_future = self._async_service.trigger_preload(self._manifest, self._adapter)

    def execute(self, phonetic_results: List[PhoneticCorrectionJobResult]) -> PhaseBatchResult:
        """
        Phase 9: Waits for the SLM model to load, then generates succinct contexts for RAG chunks sequentially.
        """
        results: List[ContextualRetrievalJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            try:
                loaded_model = self._async_service.resolve_preload(self._preload_future)
            except ModelManagerError as e:
                return PhaseBatchResult(
                    phase=PipelinePhase.CONTEXTUAL_RETRIEVAL,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading SLM model: {str(e)}"]
                )

        try:
            for item in phonetic_results:
                if not item.success:
                    continue

                source_id = item.source_id

                transcript = self._phonetic_repo.get_by_source_id(source_id)
                if not transcript:
                    err_msg = f"Cannot apply contextual retrieval: Phonetic Transcript for Source {source_id} missing."
                    results.append(
                        ContextualRetrievalJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.CONTEXTUAL_RETRIEVAL,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    # `release_after=False` retains Llama 3.2 in VRAM for the whole batch
                    contextualized_transcript = self._job_manager.run_job(
                        request,
                        self._contextualizer_api.execute,
                        transcript=transcript,
                        release_after=False
                    )

                    results.append(ContextualRetrievalJobResult(
                        source_id=source_id,
                        success=True,
                        contextual_transcript_id=contextualized_transcript.id
                    ))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    results.append(
                        ContextualRetrievalJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Contextual Retrieval for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            # Safely unload the SLM model in the background once the ENTIRE batch is done
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
            phase=PipelinePhase.CONTEXTUAL_RETRIEVAL,
            status=status,
            total_processed=len(phonetic_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )