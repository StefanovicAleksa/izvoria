# Canvas: sentence_reconstruction_phase_v2
# Path: app/orchestration/service/phases/sentence_reconstruction_phase.py

import concurrent.futures
from typing import List, Optional

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

# CRITICAL FIX: Pointing to the new feature directory (removed _old)
from app.features.sentence_reconstruction.service.reconstruction_api import SentenceReconstructionAPI
from app.features.transcription.service.transcription_api import TranscriptionAPI

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, DiarizeJobResult, SentenceReconstructionJobResult


class SentenceReconstructionPhase:
    def __init__(
            self,
            job_manager: JobManager,
            reconstruction_api: SentenceReconstructionAPI,
            transcription_api: TranscriptionAPI,
            async_model_service: AsyncModelService,
            punctuation_manifest: ModelManifest,  # Renamed from sat_manifest
            punctuation_adapter: IModelLoader     # Renamed from sat_adapter
    ):
        self._job_manager = job_manager
        self._reconstruction_api = reconstruction_api
        self._transcription_api = transcription_api
        self._async_service = async_model_service
        self._manifest = punctuation_manifest
        self._adapter = punctuation_adapter

        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        """Triggers the background loading of the Token Classification (Punctuation) model."""
        if not getattr(self._adapter, "requires_model", True):
            return  # this language keeps Whisper's own punctuation: there is nothing to load
        self._preload_future = self._async_service.trigger_preload(self._manifest, self._adapter)

    def discard_preload(self) -> None:
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._adapter)

    def execute(self, diarize_results: List[DiarizeJobResult]) -> PhaseBatchResult:
        """
        Phase 7: Waits for the punctuation model to load, then deterministically reconstructs sentences sequentially.
        """
        results: List[SentenceReconstructionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        # 1. Wait for the background GPU loading to finish
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
            except ModelManagerError as e:
                return PhaseBatchResult(
                    phase=PipelinePhase.SENTENCE_RECONSTRUCTION,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Punctuation model: {str(e)}"]
                )

        try:
            for item in diarize_results:
                if not item.success:
                    continue

                source_id = item.source_id

                # Fetch the enriched, diarized transcript to perform sentence reconstruction
                transcript = self._transcription_api.get_transcript(source_id)
                if not transcript:
                    err_msg = f"Cannot reconstruct sentences: Transcript for Source {source_id} missing."
                    results.append(SentenceReconstructionJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.SENTENCE_RECONSTRUCTION,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    # Execute API.
                    # `release_after=False` retains the Punctuation model in VRAM for the whole batch
                    reconstructed_transcript = self._job_manager.run_job(
                        request,
                        self._reconstruction_api.reconstruct_transcript,
                        transcript=transcript,
                        release_after=False
                    )

                    results.append(SentenceReconstructionJobResult(
                        source_id=source_id,
                        success=True,
                        reconstructed_transcript_id=reconstructed_transcript.id
                    ))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    results.append(SentenceReconstructionJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Sentence Reconstruction for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            # 2. Safely unload the Punctuation model in the background once the entire batch is done
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
            phase=PipelinePhase.SENTENCE_RECONSTRUCTION,
            status=status,
            total_processed=len(diarize_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )