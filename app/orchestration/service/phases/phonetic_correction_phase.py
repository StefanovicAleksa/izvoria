# Canvas: phonetic_correction_phase_v1
# Path: app/orchestration/service/phases/phonetic_correction_phase.py

import concurrent.futures
from typing import List, Optional

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.phonetic_correction.service.correction_api import PhoneticCorrectionAPI
from app.features.sentence_reconstruction.domain.interfaces import ISentenceRepository
from app.features.domain_adaptation.domain.interfaces import IDomainAdaptationRepository

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, SentenceReconstructionJobResult, PhoneticCorrectionJobResult

class PhoneticCorrectionPhase:
    def __init__(
            self,
            job_manager: JobManager,
            correction_api: PhoneticCorrectionAPI,
            sentence_repo: ISentenceRepository,
            domain_repo: IDomainAdaptationRepository,
            async_model_service: AsyncModelService,
            manifest: ModelManifest,
            adapter: IModelLoader
    ):
        self._job_manager = job_manager
        self._correction_api = correction_api
        self._sentence_repo = sentence_repo
        self._domain_repo = domain_repo
        self._async_service = async_model_service
        self._manifest = manifest
        self._adapter = adapter
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        """Triggers background loading of the Qwen-2.5 model."""
        self._preload_future = self._async_service.trigger_preload(self._manifest, self._adapter)

    def execute(self, reconstruction_results: List[SentenceReconstructionJobResult]) -> PhaseBatchResult:
        """
        Phase 8: Waits for the correction model to load, then patches anomalies.
        """
        results: List[PhoneticCorrectionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            try:
                loaded_model = self._async_service.resolve_preload(self._preload_future)
            except ModelManagerError as e:
                return PhaseBatchResult(
                    phase=PipelinePhase.PHONETIC_CORRECTION,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Phonetic model: {str(e)}"]
                )

        try:
            for item in reconstruction_results:
                if not item.success:
                    continue

                source_id = item.source_id

                transcript = self._sentence_repo.get_by_source_id(source_id)
                if not transcript:
                    err_msg = f"Cannot apply phonetic correction: Reconstructed Transcript for Source {source_id} missing."
                    results.append(PhoneticCorrectionJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                glossary = self._domain_repo.get_glossary_by_source_id(source_id)
                industry = glossary.industry if glossary else "GENERAL"
                sub_domain = glossary.sub_domain if glossary else "GENERAL"

                request = JobRequest(
                    job_type=JobType.PHONETIC_CORRECTION,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    # Note that `execute` in PhoneticCorrectionAPI has its own inference loops
                    corrected_transcript = self._job_manager.run_job(
                        request,
                        self._correction_api.execute,
                        transcript=transcript,
                        industry=industry,
                        sub_domain=sub_domain
                    )
                    results.append(PhoneticCorrectionJobResult(
                        source_id=source_id,
                        success=True,
                        phonetic_transcript_id=corrected_transcript.id
                    ))
                    success_count += 1
                except Exception as e:
                    err_msg = str(e)
                    results.append(PhoneticCorrectionJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Phonetic Correction for Source {source_id}: {err_msg}")
                    failed_count += 1

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
            phase=PipelinePhase.PHONETIC_CORRECTION,
            status=status,
            total_processed=len(reconstruction_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )