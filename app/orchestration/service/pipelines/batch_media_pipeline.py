# Canvas: batch_media_pipeline_v28
# Path: app/orchestration/service/pipelines/batch_media_pipeline.py

import uuid
import logging
from typing import Dict, List

from app.features.storage.domain.entities import SourceType
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import (
    PhaseBatchResult,
    IngestJobResult,
    AudioExtractionJobResult,
    VadJobResult,
    TranscribeJobResult,
    HallucinationCorrectionJobResult,
    DiarizeJobResult,
    SentenceReconstructionJobResult,
    DomainPhoneticCorrectionJobResult,
    ContextualRetrievalJobResult,
    VisualCueDetectionJobResult,
    FrameExtractionJobResult,
    VisualAnalysisJobResult,
    VisualHallucinationCorrectionJobResult,
)
from app.orchestration.exceptions.workflow import WorkflowStateError, WorkflowExecutionError

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.phases.vad_phase import VadPhase
from app.orchestration.service.phases.domain_adaptation_phase import DomainAdaptationPhase
from app.orchestration.service.phases.transcribe_phase import TranscribePhase
from app.orchestration.service.phases.hallucination_correction_phase import HallucinationCorrectionPhase
from app.orchestration.service.phases.diarize_phase import DiarizePhase
from app.orchestration.service.phases.sentence_reconstruction_phase import SentenceReconstructionPhase
from app.orchestration.service.phases.domain_phonetic_correction_phase import DomainPhoneticCorrectionPhase
from app.orchestration.service.phases.contextual_retrieval_phase import ContextualRetrievalPhase
from app.orchestration.service.phases.visual_cue_detection_phase import VisualCueDetectionPhase
from app.orchestration.service.phases.frame_extraction_phase import FrameExtractionPhase
from app.orchestration.service.phases.visual_analysis_phase import VisualAnalysisPhase
from app.orchestration.service.phases.visual_hallucination_correction_phase import VisualHallucinationCorrectionPhase
from app.orchestration.service.phases.visual_context_synthesis_phase import VisualContextSynthesisPhase
from app.orchestration.service.phases.context_unification_phase import ContextUnificationPhase

logger = logging.getLogger(__name__)


class BatchMediaPipeline:
    def __init__(
            self,
            scan_phase: ScanPhase,
            ingest_phase: IngestPhase,
            extract_audio_phase: ExtractAudioPhase,
            vad_phase: VadPhase,
            domain_adaptation_phase: DomainAdaptationPhase,
            transcribe_phase: TranscribePhase,
            hallucination_correction_phase: HallucinationCorrectionPhase,
            diarize_phase: DiarizePhase,
            sentence_reconstruction_phase: SentenceReconstructionPhase,
            domain_phonetic_correction_phase: DomainPhoneticCorrectionPhase,
            contextual_retrieval_phase: ContextualRetrievalPhase,
            visual_cue_detection_phase: VisualCueDetectionPhase,
            frame_extraction_phase: FrameExtractionPhase,
            visual_analysis_phase: VisualAnalysisPhase,
            visual_hallucination_correction_phase: VisualHallucinationCorrectionPhase,
            visual_context_synthesis_phase: VisualContextSynthesisPhase,
            context_unification_phase: ContextUnificationPhase
    ):
        self._scan_phase = scan_phase
        self._ingest_phase = ingest_phase
        self._extract_audio = extract_audio_phase
        self._vad_phase = vad_phase
        self._domain_adaptation_phase = domain_adaptation_phase
        self._transcribe_phase = transcribe_phase
        self._hallucination_correction_phase = hallucination_correction_phase
        self._diarize_phase = diarize_phase
        self._sentence_reconstruction_phase = sentence_reconstruction_phase
        self._domain_phonetic_correction_phase = domain_phonetic_correction_phase
        self._contextual_retrieval_phase = contextual_retrieval_phase
        self._visual_cue_detection_phase = visual_cue_detection_phase
        self._frame_extraction_phase = frame_extraction_phase
        self._visual_analysis_phase = visual_analysis_phase
        self._visual_hallucination_correction_phase = visual_hallucination_correction_phase
        self._visual_context_synthesis_phase = visual_context_synthesis_phase
        self._context_unification_phase = context_unification_phase

    def process_directory(self, target_directory: str, source_type: SourceType) -> Dict[PipelinePhase, PhaseBatchResult]:
        pipeline_results: Dict[PipelinePhase, PhaseBatchResult] = {}
        logger.info(f"========== Starting Onyx Batch Pipeline for [{target_directory}] ==========")

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
            if not video_source_ids: return pipeline_results

            # ==========================================
            # PHASE 3: AUDIO EXTRACTION (CPU BOUND)
            # ==========================================
            self._vad_phase.preload_model()
            extraction_result = self._extract_audio.execute(video_source_ids)
            pipeline_results[PipelinePhase.AUDIO_EXTRACTION] = extraction_result
            if extraction_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: Audio Extraction failed. Errors: {extraction_result.errors}")

            successful_extraction_items = [item for item in extraction_result.items if
                                           isinstance(item, AudioExtractionJobResult) and item.success]
            if not successful_extraction_items: return pipeline_results

            # ==========================================
            # PHASE 4: VOICE ACTIVITY DETECTION (GPU BOUND)
            # ==========================================
            vad_result = self._vad_phase.execute(successful_extraction_items)
            pipeline_results[PipelinePhase.VAD] = vad_result

            successful_vad_items = [item for item in vad_result.items if
                                    isinstance(item, VadJobResult) and item.success]
            if not successful_vad_items: return pipeline_results

            # ==========================================
            # PHASE 4.5: DOMAIN ADAPTATION (GPU BOUND)
            # ==========================================
            domain_result = self._domain_adaptation_phase.execute(successful_vad_items)
            pipeline_results[PipelinePhase.DOMAIN_ADAPTATION] = domain_result

            # ==========================================
            # PHASE 5: TRANSCRIPTION (GPU BOUND)
            # ==========================================
            self._transcribe_phase.preload_model()

            transcribe_result = self._transcribe_phase.execute(successful_vad_items)
            pipeline_results[PipelinePhase.TRANSCRIBE] = transcribe_result

            successful_transcribe_items: List[TranscribeJobResult] = [
                item for item in transcribe_result.items
                if isinstance(item, TranscribeJobResult) and item.success
            ]
            if not successful_transcribe_items:
                return pipeline_results

            # ==========================================
            # PHASE 5.5: HALLUCINATION CORRECTION (CPU BOUND)
            # ==========================================
            correction_result = self._hallucination_correction_phase.execute(successful_transcribe_items)
            pipeline_results[PipelinePhase.HALLUCINATION_CORRECTION] = correction_result

            successful_correction_items: List[HallucinationCorrectionJobResult] = [
                item for item in correction_result.items
                if isinstance(item, HallucinationCorrectionJobResult) and item.success
            ]
            if not successful_correction_items:
                return pipeline_results

            # ==========================================
            # PHASE 6: DIARIZATION (GPU BOUND)
            # ==========================================
            self._diarize_phase.preload_model()

            diarize_result = self._diarize_phase.execute(successful_correction_items)
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
            # PHASE 8: DOMAIN PHONETIC CORRECTION (GPU BOUND)
            # ==========================================
            self._domain_phonetic_correction_phase.preload_model()

            phonetic_result = self._domain_phonetic_correction_phase.execute(
                reconstruction_results=successful_sentence_items
            )
            pipeline_results[PipelinePhase.DOMAIN_PHONETIC_CORRECTION] = phonetic_result

            successful_phonetic_items: List[DomainPhoneticCorrectionJobResult] = [
                item for item in phonetic_result.items
                if isinstance(item, DomainPhoneticCorrectionJobResult) and item.success
            ]
            if not successful_phonetic_items:
                return pipeline_results

            # ==========================================
            # PHASE 9: CONTEXTUAL RETRIEVAL (GPU BOUND)
            # ==========================================
            self._contextual_retrieval_phase.preload_model()

            contextual_result = self._contextual_retrieval_phase.execute(successful_phonetic_items)
            pipeline_results[PipelinePhase.CONTEXTUAL_RETRIEVAL] = contextual_result

            successful_contextual_items: List[ContextualRetrievalJobResult] = [
                item for item in contextual_result.items
                if isinstance(item, ContextualRetrievalJobResult) and item.success
            ]
            if not successful_contextual_items:
                return pipeline_results

            # ==========================================
            # CONDITIONAL VISUAL PIPELINE (PHASES 10-14)
            # These cascade downward, skipping if visual data is absent.
            # ==========================================
            self._visual_cue_detection_phase.preload_model()
            visual_cue_result = self._visual_cue_detection_phase.execute(successful_contextual_items)
            pipeline_results[PipelinePhase.VISUAL_CUE_DETECTION] = visual_cue_result

            successful_cue_items: List[VisualCueDetectionJobResult] = [
                item for item in visual_cue_result.items
                if isinstance(item, VisualCueDetectionJobResult) and item.success
            ]

            if successful_cue_items:
                # PHASE 11: FRAME EXTRACTION (CPU BOUND)
                self._visual_analysis_phase.preload_model()
                extraction_result = self._frame_extraction_phase.execute(successful_cue_items)
                pipeline_results[PipelinePhase.FRAME_EXTRACTION] = extraction_result

                successful_extraction_items_frames: List[FrameExtractionJobResult] = [
                    item for item in extraction_result.items
                    if isinstance(item, FrameExtractionJobResult) and item.success
                ]

                if successful_extraction_items_frames:
                    # PHASE 12: VISUAL ANALYSIS (GPU BOUND)
                    self._visual_hallucination_correction_phase.preload_model()
                    analysis_result = self._visual_analysis_phase.execute(successful_extraction_items_frames)
                    pipeline_results[PipelinePhase.VISUAL_ANALYSIS] = analysis_result

                    successful_analysis_items: List[VisualAnalysisJobResult] = [
                        item for item in analysis_result.items
                        if isinstance(item, VisualAnalysisJobResult) and item.success
                    ]

                    if successful_analysis_items:
                        # PHASE 13: VISUAL HALLUCINATION CORRECTION (GPU BOUND)
                        self._visual_context_synthesis_phase.preload_model()
                        hallucination_result = self._visual_hallucination_correction_phase.execute(successful_analysis_items)
                        pipeline_results[PipelinePhase.VISUAL_HALLUCINATION_CORRECTION] = hallucination_result

                        successful_hallucination_items: List[VisualHallucinationCorrectionJobResult] = [
                            item for item in hallucination_result.items
                            if isinstance(item, VisualHallucinationCorrectionJobResult) and item.success
                        ]

                        if successful_hallucination_items:
                            # PHASE 14: VISUAL CONTEXT SYNTHESIS (GPU BOUND)
                            synthesis_result = self._visual_context_synthesis_phase.execute(successful_hallucination_items)
                            pipeline_results[PipelinePhase.VISUAL_CONTEXT_SYNTHESIS] = synthesis_result

            # ==========================================
            # PHASE 15: CONTEXT UNIFICATION (CPU BOUND / NO-VRAM)
            # ==========================================
            # We explicitly pass the Phase 9 successful_contextual_items.
            # This guarantees that even audio-only files (which safely bypassed 10-14) get correctly unified for RAG!
            unification_result = self._context_unification_phase.execute(successful_contextual_items)
            pipeline_results[PipelinePhase.CONTEXT_UNIFICATION] = unification_result

            logger.info(f"========== Pipeline Completed for Batch [{target_directory}] ==========")
            return pipeline_results

        except WorkflowStateError as e:
            logger.error(f"Workflow State Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Critical Pipeline Crash: {e}")
            raise WorkflowExecutionError(f"Critical Pipeline Crash: {str(e)}", original_exception=e) from e