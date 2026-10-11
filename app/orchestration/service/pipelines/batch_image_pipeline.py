# Path: app/orchestration/service/pipelines/batch_image_pipeline.py

import logging
import uuid
from pathlib import Path
from typing import Dict, List

from app.features.scanner.utils.filters import IMAGE_EXTENSIONS
from app.features.storage.domain.entities import SourceType

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import (
    PhaseBatchResult,
    IngestJobResult,
    ContextualRetrievalJobResult,
    ContextUnificationJobResult,
)
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.image_description_phase import ImageDescriptionPhase
from app.orchestration.service.phases.context_unification_phase import ContextUnificationPhase
from app.orchestration.service.phases.rag_indexing_phase import RagIndexingPhase

logger = logging.getLogger(__name__)


class BatchImagePipeline:
    """
    Scan -> ingest -> describe (vision model + OCR check, one job per image) -> unify -> index.

    Images have no speech and no document text, so none of the transcript stages run: the only models are the VLM
    (once for the whole batch) and the embedder. The VLM preloads while the files are copied into the vault.
    """

    def __init__(
            self,
            scan_phase: ScanPhase,
            ingest_phase: IngestPhase,
            image_description_phase: ImageDescriptionPhase,
            context_unification_phase: ContextUnificationPhase,
            rag_indexing_phase: RagIndexingPhase
    ):
        self._scan_phase = scan_phase
        self._ingest_phase = ingest_phase
        self._description_phase = image_description_phase
        self._context_unification_phase = context_unification_phase
        self._rag_indexing_phase = rag_indexing_phase

    def _discard_pending_preloads(self) -> None:
        """
        Guarantees no background preload outlives this run. Phases clear their own future once resolved, so this
        only touches preloads for phases that never executed (early return, halted branch, or an exception).
        """
        for phase in (self._description_phase, self._rag_indexing_phase):
            try:
                phase.discard_preload()
            except Exception as e:
                logger.error(f"Failed to discard pending preload for {type(phase).__name__}: {e}")

    def process_directory(self, target_directory: str) -> Dict[PipelinePhase, PhaseBatchResult]:
        pipeline_results: Dict[PipelinePhase, PhaseBatchResult] = {}
        logger.info(f"========== Starting Izvoria Image Batch Pipeline for [{target_directory}] ==========")

        try:
            # ==========================================
            # DIRECTORY SCAN
            # ==========================================
            scan_result = self._scan_phase.execute(target_directory)
            pipeline_results[PipelinePhase.SCAN] = scan_result
            if scan_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: Scan phase failed. Errors: {scan_result.errors}")
            if scan_result.successful_count == 0:
                return pipeline_results

            image_paths: List[str] = [p for p in scan_result.items if Path(p).suffix.lower() in IMAGE_EXTENSIONS]
            if not image_paths:
                logger.info("No image files detected in target directory. Halting pipeline.")
                return pipeline_results

            # ==========================================
            # INGESTION (CPU / DISK) -- the VLM loads meanwhile
            # ==========================================
            self._description_phase.preload_model()
            ingest_result = self._ingest_phase.execute(image_paths, SourceType.IMAGE)
            pipeline_results[PipelinePhase.INGEST] = ingest_result
            if ingest_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: Ingest phase failed entirely. Errors: {ingest_result.errors}")

            image_source_ids: List[uuid.UUID] = [
                item.source_id for item in ingest_result.items
                if isinstance(item, IngestJobResult) and item.success and item.source_id is not None
            ]
            if not image_source_ids:
                return pipeline_results

            # ==========================================
            # DESCRIPTION (GPU BOUND: VLM, plus CPU OCR)
            # ==========================================
            description_result = self._description_phase.execute(image_source_ids)
            pipeline_results[PipelinePhase.IMAGE_DESCRIPTION] = description_result

            successful_description_items = [item for item in description_result.items if
                                            isinstance(item, ContextualRetrievalJobResult) and item.success]
            if not successful_description_items:
                return pipeline_results

            # ==========================================
            # CONTEXT UNIFICATION (CPU BOUND / NO-VRAM)
            # ==========================================
            self._rag_indexing_phase.preload_model()

            unification_result = self._context_unification_phase.execute(successful_description_items)
            pipeline_results[PipelinePhase.CONTEXT_UNIFICATION] = unification_result

            successful_unification_items = [item for item in unification_result.items if
                                            isinstance(item, ContextUnificationJobResult) and item.success]
            if not successful_unification_items:
                return pipeline_results

            # ==========================================
            # RAG INDEXING (GPU BOUND)
            # ==========================================
            rag_indexing_result = self._rag_indexing_phase.execute(successful_unification_items)
            pipeline_results[PipelinePhase.RAG_INDEXING] = rag_indexing_result

            logger.info(f"========== Image Pipeline Completed for Batch [{target_directory}] ==========")
            return pipeline_results

        except WorkflowStateError as e:
            logger.error(f"Workflow State Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Critical Pipeline Crash: {e}")
            raise WorkflowExecutionError(f"Critical Pipeline Crash: {str(e)}", original_exception=e) from e
        finally:
            self._discard_pending_preloads()
