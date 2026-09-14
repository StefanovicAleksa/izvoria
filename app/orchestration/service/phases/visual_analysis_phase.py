# Canvas: visual_analysis_phase_v3
# Path: app/orchestration/service/phases/visual_analysis_phase.py

import concurrent.futures
import uuid
from typing import List, Optional

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.visual_analysis.service.analysis_api import VisualAnalysisAPI
from app.features.contextual_retrieval.domain.interfaces import IContextualRepository
from app.features.frame_extraction.domain.interfaces import IFrameExtractionRepository
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, FrameExtractionJobResult, VisualAnalysisJobResult


class VisualAnalysisPhase:
    def __init__(
            self,
            job_manager: JobManager,
            analysis_api: VisualAnalysisAPI,
            contextual_repo: IContextualRepository,
            frame_repo: IFrameExtractionRepository,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._analysis_api = analysis_api
        self._contextual_repo = contextual_repo
        self._frame_repo = frame_repo
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        """Triggers the background loading of Qwen-VL."""
        self._preload_future = self._async_service.trigger_preload(
            manifest=self._analysis_api.manifest,
            loader=self._loader
        )

    def _process_transcript_analysis(self, source_id: uuid.UUID, transcript: ContextualizedTranscript,
                                     vlm_model: LoadedModel) -> dict:
        """Helper method executed within the Job context."""
        analyzed_chunks_count = 0

        for chunk in transcript.chunks:
            if getattr(chunk, "needs_visual", False):
                extracted_frames = self._frame_repo.get_by_chunk_id(chunk.chunk_id)

                if extracted_frames and extracted_frames.frames:
                    self._analysis_api.process_chunk(
                        source_id=source_id,
                        chunk=chunk,
                        extracted_frames=extracted_frames,
                        vlm_model=vlm_model
                    )
                    analyzed_chunks_count += 1

        # Update the transcript in the database to persist the new visual descriptions
        self._contextual_repo.update(transcript)

        return {"analyzed_chunks_count": analyzed_chunks_count}

    def discard_preload(self) -> None:
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._loader)

    def execute(self, frame_extraction_results: List[FrameExtractionJobResult]) -> PhaseBatchResult:
        """
        Phase 12: Waits for the VLM model to load, then evaluates all extracted frames.
        """
        results: List[VisualAnalysisJobResult] = []
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
                    phase=PipelinePhase.VISUAL_ANALYSIS,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading VLM: {str(e)}"]
                )

        try:
            for item in frame_extraction_results:
                if not item.success:
                    continue

                source_id = item.source_id

                transcript = self._contextual_repo.get_by_source_id(source_id)
                if not transcript:
                    err_msg = f"Cannot run visual analysis: Contextual Transcript for Source {source_id} missing."
                    results.append(VisualAnalysisJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(err_msg)
                    failed_count += 1
                    continue

                request = JobRequest(
                    job_type=JobType.VISUAL_ANALYSIS,
                    source_id=source_id,
                    input_params={"video_source_id": str(source_id)}
                )

                try:
                    job_result_data = self._job_manager.run_job(
                        request,
                        self._process_transcript_analysis,
                        source_id=source_id,
                        transcript=transcript,
                        vlm_model=loaded_model
                    )

                    analyzed_chunks_count = job_result_data.get("analyzed_chunks_count", 0)

                    results.append(VisualAnalysisJobResult(
                        source_id=source_id,
                        success=True,
                        analyzed_chunks_count=analyzed_chunks_count
                    ))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    results.append(VisualAnalysisJobResult(source_id=source_id, success=False, error_message=err_msg))
                    errors.append(f"Failed Visual Analysis for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                self._async_service.trigger_unload(
                    loaded_model=loaded_model,
                    loader=self._loader,
                    flush=True
                )

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.VISUAL_ANALYSIS,
            status=status,
            total_processed=len(frame_extraction_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )