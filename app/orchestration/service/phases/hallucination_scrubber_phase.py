# Canvas: hallucination_scrubber_phase_v1
# Path: app/orchestration/service/phases/hallucination_scrubber_phase.py

import concurrent.futures
from typing import List, Optional

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.legacy_features.hallucination_scrubber.service.scrubber_api import HallucinationScrubberAPI

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, DiarizeJobResult, HallucinationScrubberJobResult


class HallucinationScrubberPhase:
    def __init__(
            self,
            job_manager: JobManager,
            scrubber_api: HallucinationScrubberAPI,
            async_model_service: AsyncModelService,
            scrubber_manifest: ModelManifest,
            scrubber_adapter: IModelLoader
    ):
        self._job_manager = job_manager
        self._scrubber_api = scrubber_api
        self._async_service = async_model_service
        self._manifest = scrubber_manifest
        self._adapter = scrubber_adapter

        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        """Triggers the background loading of the DeBERTa Zero-Shot model."""
        self._preload_future = self._async_service.trigger_preload(self._manifest, self._adapter)

    def execute(self, diarize_results: List[DiarizeJobResult]) -> PhaseBatchResult:
        """
        Phase 6.5: Waits for DeBERTa to load, then scrubs semantic hallucinations sequentially.
        """
        results: List[HallucinationScrubberJobResult] = []
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
                    phase=PipelinePhase.HALLUCINATION_SCRUBBER,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Scrubber model: {str(e)}"]
                )

        try:
            for item in diarize_results:
                if not item.success:
                    continue

                source_id = item.source_id

                request = JobRequest(
                    job_type=JobType.HALLUCINATION_SCRUBBER,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    # Execute API.
                    # `release_after=False` retains DeBERTa in VRAM for the whole batch
                    scrubbed_transcript = self._job_manager.run_job(
                        request,
                        self._scrubber_api.execute,
                        source_id=source_id,
                        release_after=False
                    )

                    results.append(HallucinationScrubberJobResult(
                        source_id=source_id,
                        success=True,
                        scrubbed_transcript_id=scrubbed_transcript.id
                    ))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    results.append(HallucinationScrubberJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Hallucination Scrubbing for Source {source_id}: {err_msg}")
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
            phase=PipelinePhase.HALLUCINATION_SCRUBBER,
            status=status,
            total_processed=len(diarize_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )