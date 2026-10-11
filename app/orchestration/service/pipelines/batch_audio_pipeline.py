# Canvas: batch_audio_pipeline_v2
# Path: app/orchestration/service/pipelines/batch_audio_pipeline.py

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

# The scanner's list: a format the scan drops never reaches this pipeline's filter, so there is one list
from app.features.scanner.utils.filters import AUDIO_EXTENSIONS
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
    ContextUnificationJobResult,
    RagIndexingJobResult,
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
from app.orchestration.service.phases.context_unification_phase import ContextUnificationPhase
from app.orchestration.service.phases.rag_indexing_phase import RagIndexingPhase
from app.orchestration.service.phases.storage_compaction_phase import StorageCompactionPhase

logger = logging.getLogger(__name__)


class BatchAudioPipeline:
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
            context_unification_phase: ContextUnificationPhase,
            rag_indexing_phase: RagIndexingPhase,
            storage_compaction_phase: Optional[StorageCompactionPhase] = None
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
        self._context_unification_phase = context_unification_phase
        self._rag_indexing_phase = rag_indexing_phase
        self._storage_compaction_phase = storage_compaction_phase

    def _discard_pending_preloads(self) -> None:
        """
        Guarantees no background preload outlives this run. Phases clear their own future
        once resolved, so this only touches preloads for phases that never executed
        (early return, halted branch, or an exception).
        """
        for phase in (self._vad_phase, self._transcribe_phase, self._diarize_phase, self._sentence_reconstruction_phase, self._domain_phonetic_correction_phase, self._contextual_retrieval_phase, self._rag_indexing_phase):
            try:
                phase.discard_preload()
            except Exception as e:
                logger.error(f"Failed to discard pending preload for {type(phase).__name__}: {e}")

    def process_directory(self, target_directory: str) -> Dict[PipelinePhase, PhaseBatchResult]:
        pipeline_results: Dict[PipelinePhase, PhaseBatchResult] = {}
        logger.info(f"========== Starting Izvoria Audio Batch Pipeline for [{target_directory}] ==========")

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
            # PHASE 2: INGESTION (AUDIO ONLY)
            # ==========================================
            raw_file_paths: List[str] = scan_result.items
            audio_file_paths: List[str] = [
                p for p in raw_file_paths if Path(p).suffix.lower() in AUDIO_EXTENSIONS
            ]

            if not audio_file_paths:
                logger.info("No audio files detected in target directory. Halting pipeline.")
                return pipeline_results

            ingest_result = self._ingest_phase.execute(audio_file_paths, SourceType.AUDIO)
            pipeline_results[PipelinePhase.INGEST] = ingest_result
            if ingest_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: Ingest phase failed entirely. Errors: {ingest_result.errors}")

            audio_source_ids: List[uuid.UUID] = [
                item.source_id for item in ingest_result.items
                if isinstance(item, IngestJobResult) and item.success and item.source_id is not None
            ]
            if not audio_source_ids:
                return pipeline_results

            # ==========================================
            # PHASE 3: AUDIO NORMALIZATION (CPU BOUND)
            # ==========================================
            # Audio files are converted to the same 16kHz mono WAV videos get, so every later
            # phase reads identical, already-resampled samples instead of re-decoding the original.
            self._vad_phase.preload_model()
            extraction_result = self._extract_audio.execute(audio_source_ids)
            pipeline_results[PipelinePhase.AUDIO_EXTRACTION] = extraction_result
            if extraction_result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(
                    f"Pipeline halted: Audio normalization failed. Errors: {extraction_result.errors}")

            successful_extraction_items = [item for item in extraction_result.items if
                                           isinstance(item, AudioExtractionJobResult) and item.success]
            if not successful_extraction_items:
                return pipeline_results

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

            successful_transcribe_items = [item for item in transcribe_result.items if
                                           isinstance(item, TranscribeJobResult) and item.success]
            if not successful_transcribe_items: return pipeline_results

            # ==========================================
            # PHASE 5.5: HALLUCINATION CORRECTION (CPU BOUND)
            # ==========================================
            correction_result = self._hallucination_correction_phase.execute(successful_transcribe_items)
            pipeline_results[PipelinePhase.HALLUCINATION_CORRECTION] = correction_result

            successful_correction_items = [item for item in correction_result.items if
                                           isinstance(item, HallucinationCorrectionJobResult) and item.success]
            if not successful_correction_items: return pipeline_results

            # ==========================================
            # PHASE 6: DIARIZATION (GPU BOUND)
            # ==========================================
            self._diarize_phase.preload_model()
            diarize_result = self._diarize_phase.execute(successful_correction_items)
            pipeline_results[PipelinePhase.DIARIZE] = diarize_result

            successful_diarize_items = [item for item in diarize_result.items if
                                        isinstance(item, DiarizeJobResult) and item.success]
            if not successful_diarize_items: return pipeline_results

            # ==========================================
            # PHASE 7: SENTENCE RECONSTRUCTION (GPU BOUND)
            # ==========================================
            self._sentence_reconstruction_phase.preload_model()
            sentence_result = self._sentence_reconstruction_phase.execute(successful_diarize_items)
            pipeline_results[PipelinePhase.SENTENCE_RECONSTRUCTION] = sentence_result

            successful_sentence_items = [item for item in sentence_result.items if
                                         isinstance(item, SentenceReconstructionJobResult) and item.success]
            if not successful_sentence_items: return pipeline_results

            # ==========================================
            # PHASE 8: DOMAIN PHONETIC CORRECTION (GPU BOUND)
            # ==========================================
            self._domain_phonetic_correction_phase.preload_model()
            phonetic_result = self._domain_phonetic_correction_phase.execute(
                reconstruction_results=successful_sentence_items)
            pipeline_results[PipelinePhase.DOMAIN_PHONETIC_CORRECTION] = phonetic_result

            successful_phonetic_items = [item for item in phonetic_result.items if
                                         isinstance(item, DomainPhoneticCorrectionJobResult) and item.success]
            if not successful_phonetic_items: return pipeline_results

            # ==========================================
            # PHASE 9: CONTEXTUAL RETRIEVAL (GPU BOUND)
            # ==========================================
            self._contextual_retrieval_phase.preload_model()
            contextual_result = self._contextual_retrieval_phase.execute(successful_phonetic_items)
            pipeline_results[PipelinePhase.CONTEXTUAL_RETRIEVAL] = contextual_result

            successful_contextual_items = [item for item in contextual_result.items if
                                           isinstance(item, ContextualRetrievalJobResult) and item.success]
            if not successful_contextual_items: return pipeline_results

            # NOTE: VISUAL PIPELINE (PHASES 10-14) IS BYPASSED FOR AUDIO SOURCES

            # ==========================================
            # PHASE 15: CONTEXT UNIFICATION (CPU BOUND / NO-VRAM)
            # ==========================================
            self._rag_indexing_phase.preload_model()

            unification_result = self._context_unification_phase.execute(successful_contextual_items)
            pipeline_results[PipelinePhase.CONTEXT_UNIFICATION] = unification_result

            successful_unification_items = [item for item in unification_result.items if
                                            isinstance(item, ContextUnificationJobResult) and item.success]
            if not successful_unification_items: return pipeline_results

            # ==========================================
            # PHASE 16: RAG INDEXING (GPU BOUND)
            # ==========================================
            rag_indexing_result = self._rag_indexing_phase.execute(successful_unification_items)
            pipeline_results[PipelinePhase.RAG_INDEXING] = rag_indexing_result

            # ==========================================
            # LAST: STORAGE COMPACTION (CPU BOUND)
            # ==========================================
            if self._storage_compaction_phase is not None and self._storage_compaction_phase.enabled():
                pipeline_results[PipelinePhase.STORAGE_COMPACTION] = self._storage_compaction_phase.execute(
                    [item for item in rag_indexing_result.items if isinstance(item, RagIndexingJobResult)])

            logger.info(f"========== Audio Pipeline Completed for Batch [{target_directory}] ==========")
            return pipeline_results

        except WorkflowStateError as e:
            logger.error(f"Workflow State Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Critical Pipeline Crash: {e}")
            raise WorkflowExecutionError(f"Critical Pipeline Crash: {str(e)}", original_exception=e) from e
        finally:
            self._discard_pending_preloads()
