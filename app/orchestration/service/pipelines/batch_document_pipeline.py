# Path: app/orchestration/service/pipelines/batch_document_pipeline.py

import logging
import uuid
from pathlib import Path
from typing import Dict, List

from app.features.storage.domain.entities import SourceType

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import (
    PhaseBatchResult,
    IngestJobResult,
    DocumentConversionJobResult,
    ContextualRetrievalJobResult,
    ContextUnificationJobResult,
)
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.document_conversion_phase import DocumentConversionPhase
from app.orchestration.service.phases.document_contextualization_phase import DocumentContextualizationPhase
from app.orchestration.service.phases.context_unification_phase import ContextUnificationPhase
from app.orchestration.service.phases.rag_indexing_phase import RagIndexingPhase

logger = logging.getLogger(__name__)

DOCUMENT_EXTENSIONS = {'.pdf', '.docx'}


class BatchDocumentPipeline:
    """
    Scan -> ingest -> convert (PDF pages / DOCX to Markdown, OCR for scans) -> structure + contextualize ->
    unify -> index. The only GPU phases are contextualization (the contextual SLM) and indexing (the embedder);
    the SLM preloads in the background while the CPU-only conversion runs.
    """

    def __init__(
            self,
            scan_phase: ScanPhase,
            ingest_phase: IngestPhase,
            document_conversion_phase: DocumentConversionPhase,
            document_contextualization_phase: DocumentContextualizationPhase,
            context_unification_phase: ContextUnificationPhase,
            rag_indexing_phase: RagIndexingPhase
    ):
        self._scan_phase = scan_phase
        self._ingest_phase = ingest_phase
        self._conversion_phase = document_conversion_phase
        self._contextualization_phase = document_contextualization_phase
        self._context_unification_phase = context_unification_phase
        self._rag_indexing_phase = rag_indexing_phase

    def _discard_pending_preloads(self) -> None:
        """
        Guarantees no background preload outlives this run. Phases clear their own future once resolved, so this
        only touches preloads for phases that never executed (early return, halted branch, or an exception).
        """
        for phase in (self._contextualization_phase, self._rag_indexing_phase):
            try:
                phase.discard_preload()
            except Exception as e:
                logger.error(f"Failed to discard pending preload for {type(phase).__name__}: {e}")

    def process_directory(self, target_directory: str) -> Dict[PipelinePhase, PhaseBatchResult]:
        pipeline_results: Dict[PipelinePhase, PhaseBatchResult] = {}
        logger.info(f"========== Starting Onyx Document Batch Pipeline for [{target_directory}] ==========")

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

            # ==========================================
            # INGESTION (DOCUMENTS ONLY)
            # ==========================================
            document_paths: List[str] = [
                p for p in scan_result.items if Path(p).suffix.lower() in DOCUMENT_EXTENSIONS
            ]
            if not document_paths:
                logger.info("No PDF or DOCX files detected in target directory. Halting pipeline.")
                return pipeline_results

            ingest_result = self._ingest_phase.execute(document_paths, SourceType.DOCUMENT)
            pipeline_results[PipelinePhase.INGEST] = ingest_result
            if ingest_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: Ingest phase failed entirely. Errors: {ingest_result.errors}")

            document_source_ids: List[uuid.UUID] = [
                item.source_id for item in ingest_result.items
                if isinstance(item, IngestJobResult) and item.success and item.source_id is not None
            ]
            if not document_source_ids:
                return pipeline_results

            # ==========================================
            # CONVERSION (CPU BOUND) -- the SLM loads meanwhile
            # ==========================================
            self._contextualization_phase.preload_model()
            conversion_result = self._conversion_phase.execute(document_source_ids)
            pipeline_results[PipelinePhase.DOCUMENT_CONVERSION] = conversion_result
            if conversion_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: Document conversion failed. Errors: {conversion_result.errors}")

            successful_conversion_items = [item for item in conversion_result.items if
                                           isinstance(item, DocumentConversionJobResult) and item.success]
            if not successful_conversion_items:
                return pipeline_results

            # ==========================================
            # STRUCTURING + CONTEXTUALIZATION (GPU BOUND)
            # ==========================================
            contextual_result = self._contextualization_phase.execute(successful_conversion_items)
            pipeline_results[PipelinePhase.DOCUMENT_CONTEXTUALIZATION] = contextual_result

            successful_contextual_items = [item for item in contextual_result.items if
                                           isinstance(item, ContextualRetrievalJobResult) and item.success]
            if not successful_contextual_items:
                return pipeline_results

            # ==========================================
            # CONTEXT UNIFICATION (CPU BOUND / NO-VRAM)
            # ==========================================
            self._rag_indexing_phase.preload_model()

            unification_result = self._context_unification_phase.execute(successful_contextual_items)
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

            logger.info(f"========== Document Pipeline Completed for Batch [{target_directory}] ==========")
            return pipeline_results

        except WorkflowStateError as e:
            logger.error(f"Workflow State Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Critical Pipeline Crash: {e}")
            raise WorkflowExecutionError(f"Critical Pipeline Crash: {str(e)}", original_exception=e) from e
        finally:
            self._discard_pending_preloads()
