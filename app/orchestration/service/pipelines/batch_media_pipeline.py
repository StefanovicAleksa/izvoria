# filename: orchestration_service_pipelines_batch_media_pipeline_v1.py
import uuid
from typing import Dict, List

from app.features.storage.domain.entities import SourceType

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, IngestJobResult
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase


class BatchMediaPipeline:
    def __init__(
            self,
            scan_phase: ScanPhase,
            ingest_phase: IngestPhase,
            extract_audio_phase: ExtractAudioPhase
    ):
        self._scan_phase = scan_phase
        self._ingest_phase = ingest_phase
        self._extract_audio = extract_audio_phase

    def process_directory(self, target_directory: str, source_type: SourceType) -> Dict[
        PipelinePhase, PhaseBatchResult]:
        """
        Executes the full media pipeline sequentially.
        Guarantees that batches are processed entirely before transitioning
        to the next phase, preserving VRAM and system stability.
        """
        pipeline_results: Dict[PipelinePhase, PhaseBatchResult] = {}

        try:
            # ==========================================
            # PHASE 1: DIRECTORY SCAN
            # ==========================================
            scan_result = self._scan_phase.execute(target_directory)
            pipeline_results[PipelinePhase.SCAN] = scan_result

            if scan_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: Scan phase failed. Errors: {scan_result.errors}")

            if scan_result.successful_count == 0:
                # Graceful exit: Nothing to process
                return pipeline_results

            # ==========================================
            # PHASE 2: INGESTION
            # ==========================================
            file_paths = scan_result.items
            ingest_result = self._ingest_phase.execute(file_paths, source_type)
            pipeline_results[PipelinePhase.INGEST] = ingest_result

            if ingest_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: Ingest phase failed entirely. Errors: {ingest_result.errors}")

            # ==========================================
            # PHASE TRANSITION: FILTER FOR AUDIO EXTRACTION
            # ==========================================
            # We only extract audio if the user told us this batch is VIDEO
            if source_type == SourceType.VIDEO:

                # Filter out failures; we only want successfully ingested Source IDs
                video_source_ids: List[uuid.UUID] = [
                    item.source_id
                    for item in ingest_result.items
                    if isinstance(item, IngestJobResult) and item.success and item.source_id
                ]

                if not video_source_ids:
                    # All ingests failed, or none returned IDs. Graceful skip.
                    return pipeline_results

                # ==========================================
                # PHASE 3: AUDIO EXTRACTION
                # ==========================================
                extraction_result = self._extract_audio.execute(video_source_ids)
                pipeline_results[PipelinePhase.AUDIO_EXTRACTION] = extraction_result

                if extraction_result.status == PhaseStatus.FAILED:
                    # We don't necessarily halt the app here, but we record the catastrophic failure
                    # In the future, Phase 4 (Transcribe) would check this before starting
                    pass

            return pipeline_results

        except WorkflowStateError as e:
            # Expected state machine violations (e.g., trying to proceed after a phase fails)
            raise e
        except Exception as e:
            # Unexpected pipeline crashes
            raise WorkflowExecutionError(f"Critical Pipeline Crash: {str(e)}", original_exception=e) from e