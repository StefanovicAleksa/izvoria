# Canvas: batch_media_pipeline_v7
# Path: app/orchestration/service/pipelines/batch_media_pipeline.py

import uuid
from typing import Dict, List

from app.features.storage.domain.entities import SourceType

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import (
    PhaseBatchResult, IngestJobResult, VadJobResult,
    TranscribeJobResult, DiarizeJobResult
)
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.phases.vad_phase import VadPhase
from app.orchestration.service.phases.transcribe_phase import TranscribePhase
from app.orchestration.service.phases.diarize_phase import DiarizePhase
from app.orchestration.service.phases.sentence_reconstruction_phase import SentenceReconstructionPhase


class BatchMediaPipeline:
    def __init__(
            self,
            scan_phase: ScanPhase,
            ingest_phase: IngestPhase,
            extract_audio_phase: ExtractAudioPhase,
            vad_phase: VadPhase,
            transcribe_phase: TranscribePhase,
            diarize_phase: DiarizePhase,
            sentence_reconstruction_phase: SentenceReconstructionPhase
    ):
        self._scan_phase = scan_phase
        self._ingest_phase = ingest_phase
        self._extract_audio = extract_audio_phase
        self._vad_phase = vad_phase
        self._transcribe_phase = transcribe_phase
        self._diarize_phase = diarize_phase
        self._sentence_reconstruction_phase = sentence_reconstruction_phase

    def process_directory(self, target_directory: str, source_type: SourceType) -> Dict[PipelinePhase, PhaseBatchResult]:
        pipeline_results: Dict[PipelinePhase, PhaseBatchResult] = {}

        try:
            # PHASE 1: DIRECTORY SCAN
            scan_result = self._scan_phase.execute(target_directory)
            pipeline_results[PipelinePhase.SCAN] = scan_result
            if scan_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: Scan phase failed. Errors: {scan_result.errors}")
            if scan_result.successful_count == 0:
                return pipeline_results

            # PHASE 2: INGESTION
            file_paths = scan_result.items
            ingest_result = self._ingest_phase.execute(file_paths, source_type)
            pipeline_results[PipelinePhase.INGEST] = ingest_result
            if ingest_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: Ingest phase failed entirely. Errors: {ingest_result.errors}")

            # PHASE TRANSITION: FILTER FOR AUDIO EXTRACTION
            if source_type == SourceType.VIDEO:
                video_source_ids: List[uuid.UUID] = [
                    item.source_id for item in ingest_result.items
                    if isinstance(item, IngestJobResult) and item.success and item.source_id
                ]
                if not video_source_ids:
                    return pipeline_results

                # PRELOAD TRIGGER: Load VAD model onto GPU while FFmpeg runs on CPU
                self._vad_phase.preload_model()

                # PHASE 3: AUDIO EXTRACTION (CPU BOUND)
                extraction_result = self._extract_audio.execute(video_source_ids)
                pipeline_results[PipelinePhase.AUDIO_EXTRACTION] = extraction_result
                if extraction_result.status == PhaseStatus.FAILED:
                    raise WorkflowStateError(
                        f"Pipeline halted: Audio Extraction failed entirely. Errors: {extraction_result.errors}")

                # PHASE 4: VOICE ACTIVITY DETECTION (GPU BOUND)
                vad_result = self._vad_phase.execute(extraction_result.items)
                pipeline_results[PipelinePhase.VAD] = vad_result
                if vad_result.status == PhaseStatus.FAILED:
                    pass  # Log failure, but continue

                # PHASE TRANSITION: FILTER FOR TRANSCRIPTION
                successful_vad_items = [
                    item for item in vad_result.items
                    if isinstance(item, VadJobResult) and item.success
                ]
                if not successful_vad_items:
                    return pipeline_results

                # PRELOAD TRIGGER: Load Whisper model into GPU while VAD wrap-ups finish
                self._transcribe_phase.preload_model()

                # PHASE 5: TRANSCRIPTION (GPU BOUND)
                transcribe_result = self._transcribe_phase.execute(successful_vad_items)
                pipeline_results[PipelinePhase.TRANSCRIBE] = transcribe_result
                if transcribe_result.status == PhaseStatus.FAILED:
                    pass

                # PHASE TRANSITION: FILTER FOR DIARIZATION
                successful_transcribe_items = [
                    item for item in transcribe_result.items
                    if isinstance(item, TranscribeJobResult) and item.success
                ]
                if not successful_transcribe_items:
                    return pipeline_results

                # PRELOAD TRIGGER: Load TitaNet model into GPU
                self._diarize_phase.preload_model()

                # PHASE 6: DIARIZATION (GPU BOUND)
                diarize_result = self._diarize_phase.execute(successful_transcribe_items)
                pipeline_results[PipelinePhase.DIARIZE] = diarize_result

                # PHASE TRANSITION: FILTER FOR SENTENCE RECONSTRUCTION
                successful_diarize_items = [
                    item for item in diarize_result.items
                    if isinstance(item, DiarizeJobResult) and item.success
                ]
                if not successful_diarize_items:
                    return pipeline_results

                # PRELOAD TRIGGER: Load SaT model into GPU
                self._sentence_reconstruction_phase.preload_model()

                # PHASE 7: SENTENCE RECONSTRUCTION (GPU BOUND)
                sentence_result = self._sentence_reconstruction_phase.execute(successful_diarize_items)
                pipeline_results[PipelinePhase.SENTENCE_RECONSTRUCTION] = sentence_result

            return pipeline_results

        except WorkflowStateError as e:
            raise e
        except Exception as e:
            raise WorkflowExecutionError(f"Critical Pipeline Crash: {str(e)}", original_exception=e) from e