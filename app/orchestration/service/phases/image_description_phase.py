# Path: app/orchestration/service/phases/image_description_phase.py

import concurrent.futures
import logging
import uuid
from typing import List, Optional

from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType
from app.core.jobs.service.manager import JobManager
from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.features.contextual_retrieval.service.contextualizer_api import ContextualizerAPI
from app.features.image_description.domain.entities import ImageDescription
from app.features.image_description.service.image_description_api import ImageDescriptionAPI
from app.features.visual_analysis.service.analysis_api import VisualAnalysisAPI
from app.orchestration.domain.entities import ImageDescriptionJobResult, PhaseBatchResult
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase

logger = logging.getLogger(__name__)


class ImageDescriptionPhase:
    """
    Describes each ingested image with the vision model, checks the text it read against OCR, and stores the result
    as a one-chunk contextualized transcript -- all inside one job per image, since only the VLM call is slow.

    The VLM is loaded once for the whole batch (preload / resolve / unload, like VisualAnalysisPhase) and OCR's CPU
    models are released at the end. The results are ContextualRetrievalJobResults, so ContextUnificationPhase and
    RagIndexingPhase take them unchanged.
    """

    def __init__(
            self,
            job_manager: JobManager,
            image_api: ImageDescriptionAPI,
            visual_api: VisualAnalysisAPI,
            contextualizer_api: ContextualizerAPI,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._image_api = image_api
        self._visual_api = visual_api
        self._contextualizer_api = contextualizer_api
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        # A second call before execute() would orphan the first future (see CLAUDE.md, "An unresolved preload")
        self.discard_preload()
        logger.info("Triggering background preload for the vision model (image description)...")
        self._preload_future = self._async_service.trigger_preload(
            manifest=self._visual_api.manifest, loader=self._loader
        )

    def discard_preload(self) -> None:
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._loader)

    def execute(self, source_ids: List[uuid.UUID]) -> PhaseBatchResult:
        results: List[ImageDescriptionJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0

        if not self._preload_future:
            # The VLM is only ever handed in by a resolved preload: describing an image without one would mean
            # acquiring the model per image
            self.preload_model()
        future, self._preload_future = self._preload_future, None
        try:
            loaded_model: LoadedModel = self._async_service.resolve_preload(future)
            logger.info("Successfully resolved preloaded vision model.")
        except ModelManagerError as e:
            logger.error(f"Failed to load the vision model: {e}")
            return PhaseBatchResult(
                phase=PipelinePhase.IMAGE_DESCRIPTION,
                status=PhaseStatus.FAILED,
                errors=[f"Critical failure preloading the vision model: {str(e)}"]
            )

        try:
            for source_id in source_ids:
                request = JobRequest(
                    job_type=JobType.IMAGE_DESCRIPTION,
                    source_id=source_id,
                    input_params={"image_source_id": str(source_id)}
                )
                try:
                    logger.info(f"Processing Image Description for Source: {source_id}")
                    outcome = self._job_manager.run_job(
                        request, self._describe_and_contextualize, source_id=source_id, vlm_model=loaded_model
                    )
                    results.append(ImageDescriptionJobResult(
                        source_id=source_id,
                        success=True,
                        contextual_transcript_id=outcome["_contextual_transcript_id"],
                        ocr_checked=outcome["ocr_checked"],
                        text_items_dropped=outcome["text_items_dropped"],
                        ocr_incomplete=outcome["ocr_incomplete"],
                        ocr_lines_withheld=outcome["ocr_lines_withheld"],
                    ))
                    success_count += 1
                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Source {source_id}: {err_msg}")
                    results.append(ImageDescriptionJobResult(source_id=source_id, success=False,
                                                             error_message=err_msg))
                    errors.append(f"Failed Image Description for Source {source_id}: {err_msg}")
                    failed_count += 1

        finally:
            logger.info("Unloading the vision model to free VRAM...")
            self._async_service.trigger_unload(loaded_model=loaded_model, loader=self._loader, flush=True)
            self._image_api.release()

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.IMAGE_DESCRIPTION,
            status=status,
            total_processed=len(source_ids),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )

    def _describe_and_contextualize(self, source_id: uuid.UUID, vlm_model: LoadedModel) -> dict:
        """The job's result: the scalars are recorded on the job row, "_" keys only travel back to execute()."""
        description: ImageDescription = self._image_api.execute(source_id, vlm_model)
        transcript = self._contextualizer_api.execute_image(
            source_id,
            excerpt=self._image_api.render(description),
            context_string=self._image_api.render_context(description),
        )
        return {
            "_contextual_transcript_id": transcript.id,
            "image_description_id": str(description.id),
            "ocr_checked": description.ocr_checked,
            "text_items_dropped": len(description.dropped_items),
            "ocr_incomplete": description.ocr_incomplete,
            "ocr_lines_withheld": len(description.ocr_withheld_lines),
        }
