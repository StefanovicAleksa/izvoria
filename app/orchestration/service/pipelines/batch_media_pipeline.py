# Canvas: batch_media_pipeline_v13
# Path: app/orchestration/service/pipelines/batch_media_pipeline.py

import uuid
from typing import Dict, List

from app.features.storage.domain.entities import SourceType
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import (
    PhaseBatchResult,
    IngestJobResult,
    AudioExtractionJobResult,
    VadJobResult,
    TranscribeJobResult,
    DiarizeJobResult,
    SentenceReconstructionJobResult
)
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.phases.vad_phase import VadPhase
from app.orchestration.service.phases.domain_adaptation_phase import DomainAdaptationPhase
from app.orchestration.service.phases.transcribe_phase import TranscribePhase
from app.orchestration.service.phases.diarize_phase import DiarizePhase
from app.orchestration.service.phases.sentence_reconstruction_phase import SentenceReconstructionPhase
from app.orchestration.service.phases.phonetic_correction_phase import PhoneticCorrectionPhase


class BatchMediaPipeline:
    def __init__(
            self,
            scan_phase: ScanPhase,
            ingest_phase: IngestPhase,
            extract_audio_phase: ExtractAudioPhase,
            vad_phase: VadPhase,
            domain_adaptation_phase: DomainAdaptationPhase,
            transcribe_phase: TranscribePhase,
            diarize_phase: DiarizePhase,
            sentence_reconstruction_phase: SentenceReconstructionPhase,
            phonetic_correction_phase: PhoneticCorrectionPhase  # <-- THIS makes PyCharm happy
    ):
        self._scan_phase = scan_phase
        self._ingest_phase = ingest_phase
        self._extract_audio = extract_audio_phase
        self._vad_phase = vad_phase
        self._domain_adaptation_phase = domain_adaptation_phase
        self._transcribe_phase = transcribe_phase
        self._diarize_phase = diarize_phase
        self._sentence_reconstruction_phase = sentence_reconstruction_phase
        self._phonetic_correction_phase = phonetic_correction_phase

    def process_directory(self, target_directory: str, source_type: SourceType) -> Dict[
        PipelinePhase, PhaseBatchResult]:
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
                return pipeline_results

            # ==========================================
            # PHASE 2: INGESTION
            # ==========================================
            file_paths: List[str] = scan_result.items
            ingest_result = self._ingest_phase.execute(file_paths, source_type)
            pipeline_results[PipelinePhase.INGEST] = ingest_result

            if ingest_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: Ingest phase failed entirely. Errors: {ingest_result.errors}")

            video_source_ids: List[uuid.UUID] = [
                item.source_id for item in ingest_result.items
                if isinstance(item, IngestJobResult) and item.success and item.source_id
            ]

            if not video_source_ids:
                return pipeline_results

            # ==========================================
            # PHASE 3: AUDIO EXTRACTION (CPU BOUND)
            # ==========================================
            self._vad_phase.preload_model()

            extraction_result = self._extract_audio.execute(video_source_ids)
            pipeline_results[PipelinePhase.AUDIO_EXTRACTION] = extraction_result

            if extraction_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: Audio Extraction failed entirely. Errors: {extraction_result.errors}")

            successful_extraction_items: List[AudioExtractionJobResult] = [
                item for item in extraction_result.items
                if isinstance(item, AudioExtractionJobResult) and item.success
            ]
            if not successful_extraction_items:
                return pipeline_results

            # ==========================================
            # PHASE 4: VOICE ACTIVITY DETECTION (GPU BOUND)
            # ==========================================
            vad_result = self._vad_phase.execute(successful_extraction_items)
            pipeline_results[PipelinePhase.VAD] = vad_result

            successful_vad_items: List[VadJobResult] = [
                item for item in vad_result.items
                if isinstance(item, VadJobResult) and item.success
            ]
            if not successful_vad_items:
                # If VAD fails completely, there is no audio to transcribe. Halt.
                return pipeline_results

            # ==========================================
            # PHASE 4.5: DOMAIN ADAPTATION (GPU BOUND)
            # ==========================================
            domain_result = self._domain_adaptation_phase.execute(successful_vad_items)
            pipeline_results[PipelinePhase.DOMAIN_ADAPTATION] = domain_result

            if domain_result.status == PhaseStatus.FAILED:
                # NON-FATAL ERROR: If we fail to build a custom glossary, we tolerate the failure
                # and proceed to standard transcription without domain-specific hints.
                pass

            # ==========================================
            # PHASE 5: TRANSCRIPTION (GPU BOUND)
            # ==========================================
            self._transcribe_phase.preload_model()

            # Transcription relies on VAD output, fetching the glossary from the DB automatically.
            transcribe_result = self._transcribe_phase.execute(successful_vad_items)
            pipeline_results[PipelinePhase.TRANSCRIBE] = transcribe_result

            successful_transcribe_items: List[TranscribeJobResult] = [
                item for item in transcribe_result.items
                if isinstance(item, TranscribeJobResult) and item.success
            ]
            if not successful_transcribe_items:
                # If transcription fails, subsequent NLP tasks cannot proceed. Halt.
                return pipeline_results

            # ==========================================
            # PHASE 6: DIARIZATION (GPU BOUND)
            # ==========================================
            self._diarize_phase.preload_model()

            diarize_result = self._diarize_phase.execute(successful_transcribe_items)
            pipeline_results[PipelinePhase.DIARIZE] = diarize_result

            successful_diarize_items: List[DiarizeJobResult] = [
                item for item in diarize_result.items
                if isinstance(item, DiarizeJobResult) and item.success
            ]
            if not successful_diarize_items:
                return pipeline_results

            # ==========================================
            # PHASE 7: SENTENCE RECONSTRUCTION (GPU BOUND)
            # ==========================================
            self._sentence_reconstruction_phase.preload_model()

            sentence_result = self._sentence_reconstruction_phase.execute(successful_diarize_items)
            pipeline_results[PipelinePhase.SENTENCE_RECONSTRUCTION] = sentence_result

            successful_sentence_items: List[SentenceReconstructionJobResult] = [
                item for item in sentence_result.items
                if isinstance(item, SentenceReconstructionJobResult) and item.success
            ]
            if not successful_sentence_items:
                return pipeline_results

            # ==========================================
            # PHASE 8: PHONETIC CORRECTION (GPU BOUND)
            # ==========================================
            self._phonetic_correction_phase.preload_model()

            phonetic_result = self._phonetic_correction_phase.execute(successful_sentence_items)
            pipeline_results[PipelinePhase.PHONETIC_CORRECTION] = phonetic_result

            # End of pipeline. Returning the full result dictionary containing all phase statuses and errors.
            return pipeline_results

        except WorkflowStateError as e:
            raise e
        except Exception as e:
            raise WorkflowExecutionError(f"Critical Pipeline Crash: {str(e)}", original_exception=e) from e