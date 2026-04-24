# Canvas: main_v32
# Path: app/main.py

import argparse
import sys
import os
from pathlib import Path

# CRITICAL FIX: Ensure PyTorch uses expandable segments to prevent fragmentation crashes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from app.core.config import settings
from app.core.database.service.manager import db_manager
from app.core.database.data.engine import Base

from app.core.jobs.data.repositories import JobRepository
from app.core.jobs.service.manager import JobManager
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.storage.data.repositories import StorageRepository
from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import SourceType
from app.features.scanner.service.scanner import FileScanner
from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter
from app.features.audio_extractor.data.repositories import AudioExtractionRepository
from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI

from app.features.vad.data.nemo_adapter import NeMoVadAdapter
from app.features.vad.data.repositories import VadRepository
from app.features.vad.service.vad_api import VadAPI

from app.features.domain_adaptation.data.repositories import DomainAdaptationRepository
from app.features.domain_adaptation.data.scout_adapter import ScoutWhisperAdapter
from app.features.domain_adaptation.data.llama_adapter import LlamaAdapter
from app.features.domain_adaptation.utils.smart_sampler import SmartSampler
from app.features.domain_adaptation.service.adaptation_api import DomainAdaptationAPI

from app.features.transcription.data.repositories import TranscriptRepository
from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
from app.features.transcription.service.transcription_api import TranscriptionAPI

from app.features.hallucination_correction.service.correction_api import CorrectionAPI

from app.features.diarization.data.repositories import DiarizationRepository
from app.features.diarization.data.nemo_adapter import NeMoDiarizationAdapter
from app.features.diarization.service.diarization_api import DiarizationAPI

from app.features.sentence_reconstruction.data.repositories import PostgresSentenceRepository
from app.features.sentence_reconstruction.data.punctuation_adapter import PunctuationAdapter
from app.features.sentence_reconstruction.service.reconstruction_api import SentenceReconstructionAPI

from app.features.domain_phonetic_correction.data.repositories import DomainPhoneticRepository
from app.features.domain_phonetic_correction.data.qwen_adapter import QwenDomainAdapter
from app.features.domain_phonetic_correction.service.correction_api import DomainPhoneticCorrectionAPI

from app.features.contextual_retrieval.data.repositories import ContextualRepository
from app.features.contextual_retrieval.data.slm_adapter import SLMContextAdapter
from app.features.contextual_retrieval.service.contextualizer_api import ContextualizerAPI

from app.features.visual_cue_detection.data.qwen_detector_adapter import QwenDetectorAdapter
from app.features.visual_cue_detection.service.detection_api import VisualCueDetectionAPI

from app.features.frame_extraction.data.ffmpeg_adapter import FFmpegExtractionAdapter
from app.features.frame_extraction.data.repositories import FrameExtractionRepository
from app.features.frame_extraction.service.extraction_api import FrameExtractionAPI
from app.features.frame_extraction.utils.quality_filter import QualityFilter
from app.features.frame_extraction.utils.frame_deduplicator import FrameDeduplicator

from app.features.visual_analysis.data.vlm_adapter import QwenVLMAdapter
from app.features.visual_analysis.data.repositories import VisualAnalysisRepository
from app.features.visual_analysis.service.analysis_api import VisualAnalysisAPI

# Phase 13 Integration Imports
from app.features.visual_hallucination_correction.data.slm_adapter import QwenHallucinationAdapter
from app.features.visual_hallucination_correction.data.repositories import VisualHallucinationRepository
from app.features.visual_hallucination_correction.service.correction_api import VisualHallucinationCorrectionAPI

# Phase 14 Integration Imports
from app.features.visual_context_synthesis.data.slm_adapter import VisualSLMAdapter
from app.features.visual_context_synthesis.data.repositories import VisualSynthesisRepository
from app.features.visual_context_synthesis.service.synthesis_api import VisualSynthesisAPI

# Phase 15 Integration Imports
from app.features.context_unification.data.repositories import ContextUnificationRepository
from app.features.context_unification.service.unification_api import UnificationAPI

# Phase 15.5 Integration Imports
from app.features.clip_extractor.data.ffmpeg_video_adapter import FFmpegVideoAdapter
from app.features.clip_extractor.data.repositories import VideoClipRepository
from app.features.clip_extractor.service.extractor_api import ClipExtractorAPI

# Phase 16 Integration Imports
from app.features.rag_storage.data.postgres_rag_repo import PostgresRagRepository
from app.features.rag_storage.data.hf_embedder import HuggingFaceEmbeddingAdapter
from app.features.rag_storage.service.storage_api import RAGStorageService

from app.orchestration.domain.enums import PipelinePhase
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
from app.orchestration.service.phases.clip_extraction_phase import ClipExtractionPhase
from app.orchestration.service.phases.rag_indexing_phase import RagIndexingPhase
from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline


def bootstrap_database() -> None:
    try:
        with db_manager.session() as session:
            engine = session.get_bind()
            Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[FATAL] Could not connect to the database. Ensure Postgres is running.\nError: {e}")
        sys.exit(1)


def get_orchestration_pipeline() -> BatchMediaPipeline:
    # 1. Core Services
    job_repo = JobRepository(db_manager)
    job_manager = JobManager(job_repo)
    vram_monitor = PyTorchVramMonitor()
    model_orchestrator = ModelManagerOrchestrator(vram_monitor)

    bg_worker = BackgroundModelWorker(model_orchestrator)
    async_model_service = AsyncModelService(bg_worker)

    # 2. Storage & Extraction
    storage_repo = StorageRepository(db_manager)
    storage_api = StorageAPI(storage_repo)
    storage_api.file_manager.base_path.mkdir(parents=True, exist_ok=True)

    scanner = FileScanner()
    audio_repo = AudioExtractionRepository(db_manager)
    ffmpeg_adapter = FFmpegAdapter()
    audio_extractor = AudioExtractorAPI(ffmpeg_adapter, audio_repo, storage_api)

    # 3. Voice Activity Detection
    vad_repo = VadRepository(db_manager)
    nemo_adapter = NeMoVadAdapter()
    vad_api = VadAPI(vad_repo, model_orchestrator, nemo_adapter)

    vad_manifest = ModelManifest(
        model_type=ModelType.VAD,
        model_path=settings.VAD_MODEL_PATH,
        required_vram_mb=settings.VAD_VRAM_REQUIREMENT_MB
    )

    # 4. Domain Adaptation
    domain_repo = DomainAdaptationRepository(db_manager)
    scout_adapter = ScoutWhisperAdapter()
    llama_adapter = LlamaAdapter()
    smart_sampler = SmartSampler(max_speech_per_zone_sec=120.0, min_duration_sec=2.0)

    domain_adaptation_api = DomainAdaptationAPI(
        repository=domain_repo,
        orchestrator=model_orchestrator,
        scout_adapter=scout_adapter,
        llama_adapter=llama_adapter,
        smart_sampler=smart_sampler
    )

    # 5. Transcription
    transcript_repo = TranscriptRepository(db_manager)
    whisper_adapter = FasterWhisperAdapter()
    transcription_api = TranscriptionAPI(
        repository=transcript_repo,
        vad_repository=vad_repo,
        orchestrator=model_orchestrator,
        adapter=whisper_adapter
    )

    transcribe_manifest = ModelManifest(
        model_type=ModelType.TRANSCRIPTION,
        model_path=settings.WHISPER_MODEL_PATH,
        required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
    )

    # 5.5 Hallucination Correction
    correction_api = CorrectionAPI(
        transcript_repository=transcript_repo,
        vad_repository=vad_repo
    )

    # 6. Diarization
    diarize_repo = DiarizationRepository(db_manager)
    diarize_adapter = NeMoDiarizationAdapter()
    diarization_api = DiarizationAPI(
        repository=diarize_repo,
        vad_repository=vad_repo,
        transcription_api=transcription_api,
        storage_api=storage_api,
        orchestrator=model_orchestrator,
        adapter=diarize_adapter
    )

    diarize_manifest = ModelManifest(
        model_type=ModelType.DIARIZATION,
        model_path=settings.DIARIZATION_MODEL_PATH,
        required_vram_mb=settings.DIARIZATION_VRAM_REQUIREMENT_MB
    )

    # 7. Sentence Reconstruction
    sentence_repo = PostgresSentenceRepository(db_manager)
    punctuation_adapter = PunctuationAdapter()
    reconstruction_api = SentenceReconstructionAPI(
        repository=sentence_repo,
        orchestrator=model_orchestrator,
        adapter=punctuation_adapter
    )

    punctuation_manifest = ModelManifest(
        model_type=ModelType.SENTENCE_RECONSTRUCTION,
        model_path=settings.PUNCTUATION_MODEL_PATH,
        required_vram_mb=settings.PUNCTUATION_VRAM_REQUIREMENT_MB
    )

    # 8. Domain Phonetic Correction
    domain_phonetic_repo = DomainPhoneticRepository(db_manager)
    domain_phonetic_adapter = QwenDomainAdapter()
    domain_phonetic_correction_api = DomainPhoneticCorrectionAPI(
        repository=domain_phonetic_repo,
        orchestrator=model_orchestrator,
        patch_adapter=domain_phonetic_adapter
    )

    domain_phonetic_manifest = ModelManifest(
        model_type=ModelType.DOMAIN_PHONETIC_CORRECTION,
        model_path=settings.DOMAIN_PHONETIC_CORRECTION_MODEL_PATH,
        required_vram_mb=settings.DOMAIN_PHONETIC_CORRECTION_VRAM_REQUIREMENT_MB
    )

    # 9. Contextual Retrieval
    contextual_repo = ContextualRepository(db_manager)
    slm_adapter = SLMContextAdapter()
    contextualizer_api = ContextualizerAPI(
        repository=contextual_repo,
        orchestrator=model_orchestrator,
        slm_adapter=slm_adapter
    )

    # 10. Visual Cue Detection
    cue_detector_adapter = QwenDetectorAdapter()
    visual_cue_api = VisualCueDetectionAPI(
        orchestrator=model_orchestrator,
        adapter=cue_detector_adapter,
        repository=contextual_repo
    )

    # 11. Frame Extraction
    frame_repo = FrameExtractionRepository(db_manager)
    ffmpeg_ext_adapter = FFmpegExtractionAdapter()
    quality_filter = QualityFilter()
    deduplicator = FrameDeduplicator()
    frame_extraction_api = FrameExtractionAPI(
        repository=frame_repo,
        ffmpeg_adapter=ffmpeg_ext_adapter,
        storage_api=storage_api,
        quality_filter=quality_filter,
        deduplicator=deduplicator
    )

    # 12. Visual Analysis
    visual_analysis_repo = VisualAnalysisRepository(db_manager)
    vlm_adapter = QwenVLMAdapter()
    visual_analysis_api = VisualAnalysisAPI(
        repository=visual_analysis_repo,
        vlm_adapter=vlm_adapter,
        storage_api=storage_api
    )

    # 13. Visual Hallucination Correction
    visual_hallucination_repo = VisualHallucinationRepository(db_manager)
    visual_hallucination_adapter = QwenHallucinationAdapter()
    visual_hallucination_api = VisualHallucinationCorrectionAPI(
        repository=visual_hallucination_repo,
        adapter=visual_hallucination_adapter,
        contextual_repo=contextual_repo,
        visual_analysis_repo=visual_analysis_repo,
        orchestrator=model_orchestrator
    )

    # 14. Phase 14: Visual Context Synthesis
    visual_synthesis_repo = VisualSynthesisRepository(db_manager)
    visual_synthesis_adapter = VisualSLMAdapter()
    visual_synthesis_api = VisualSynthesisAPI(
        repository=visual_synthesis_repo,
        orchestrator=model_orchestrator,
        slm_adapter=visual_synthesis_adapter,
        visual_analysis_repo=visual_analysis_repo,
        hallucination_repo=visual_hallucination_repo
    )

    # 15. Context Unification
    context_unification_repo = ContextUnificationRepository(db_manager)
    unification_api = UnificationAPI(repository=context_unification_repo)

    # 15.5 Clip Extraction
    clip_repo = VideoClipRepository(db_manager)
    ffmpeg_video_adapter = FFmpegVideoAdapter(timeout_seconds=300)
    clip_extractor_api = ClipExtractorAPI(
        ffmpeg_adapter=ffmpeg_video_adapter,
        repository=clip_repo,
        storage_api=storage_api
    )

    # 16. RAG Indexing
    rag_repo = PostgresRagRepository()
    hf_embedder = HuggingFaceEmbeddingAdapter()
    rag_storage_service = RAGStorageService(
        rag_repo=rag_repo,
        orchestrator=model_orchestrator,
        embedder=hf_embedder
    )

    # --- Orchestration Phases ---
    scan_phase = ScanPhase(job_manager, scanner)
    ingest_phase = IngestPhase(job_manager, storage_api)
    extract_audio_phase = ExtractAudioPhase(job_manager, audio_extractor)

    vad_phase = VadPhase(
        job_manager=job_manager,
        vad_api=vad_api,
        storage_api=storage_api,
        async_model_service=async_model_service,
        vad_manifest=vad_manifest,
        vad_adapter=nemo_adapter
    )

    domain_adaptation_phase = DomainAdaptationPhase(
        job_manager=job_manager,
        adaptation_api=domain_adaptation_api,
        storage_api=storage_api,
        vad_repo=vad_repo
    )

    transcribe_phase = TranscribePhase(
        job_manager=job_manager,
        transcription_api=transcription_api,
        storage_api=storage_api,
        domain_repo=domain_repo,
        async_model_service=async_model_service,
        transcribe_manifest=transcribe_manifest,
        transcribe_adapter=whisper_adapter
    )

    hallucination_correction_phase = HallucinationCorrectionPhase(
        job_manager=job_manager,
        correction_api=correction_api
    )

    diarize_phase = DiarizePhase(
        job_manager=job_manager,
        diarization_api=diarization_api,
        storage_api=storage_api,
        async_model_service=async_model_service,
        diarize_manifest=diarize_manifest,
        diarize_adapter=diarize_adapter
    )

    sentence_reconstruction_phase = SentenceReconstructionPhase(
        job_manager=job_manager,
        reconstruction_api=reconstruction_api,
        transcription_api=transcription_api,
        async_model_service=async_model_service,
        punctuation_manifest=punctuation_manifest,
        punctuation_adapter=punctuation_adapter
    )

    domain_phonetic_correction_phase = DomainPhoneticCorrectionPhase(
        job_manager=job_manager,
        correction_api=domain_phonetic_correction_api,
        sentence_repo=sentence_repo,
        domain_repo=domain_repo,
        async_model_service=async_model_service,
        manifest=domain_phonetic_manifest,
        adapter=domain_phonetic_adapter
    )

    contextual_retrieval_phase = ContextualRetrievalPhase(
        job_manager=job_manager,
        contextualizer_api=contextualizer_api,
        phonetic_repo=domain_phonetic_repo,
        async_model_service=async_model_service,
        loader=slm_adapter
    )

    visual_cue_detection_phase = VisualCueDetectionPhase(
        job_manager=job_manager,
        detection_api=visual_cue_api,
        contextual_repo=contextual_repo,
        async_model_service=async_model_service,
        loader=cue_detector_adapter
    )

    frame_extraction_phase = FrameExtractionPhase(
        job_manager=job_manager,
        extraction_api=frame_extraction_api,
        contextual_repo=contextual_repo,
        storage_api=storage_api
    )

    visual_analysis_phase = VisualAnalysisPhase(
        job_manager=job_manager,
        analysis_api=visual_analysis_api,
        contextual_repo=contextual_repo,
        frame_repo=frame_repo,
        async_model_service=async_model_service,
        loader=vlm_adapter
    )

    visual_hallucination_correction_phase = VisualHallucinationCorrectionPhase(
        job_manager=job_manager,
        correction_api=visual_hallucination_api,
        async_model_service=async_model_service,
        loader=visual_hallucination_adapter
    )

    visual_context_synthesis_phase = VisualContextSynthesisPhase(
        job_manager=job_manager,
        synthesis_api=visual_synthesis_api,
        contextual_repo=contextual_repo,
        async_model_service=async_model_service,
        loader=visual_synthesis_adapter
    )

    context_unification_phase = ContextUnificationPhase(
        job_manager=job_manager,
        unification_api=unification_api,
        contextual_repo=contextual_repo,
        visual_synthesis_repo=visual_synthesis_repo
    )

    clip_extraction_phase = ClipExtractionPhase(
        job_manager=job_manager,
        extractor_api=clip_extractor_api,
        unified_repo=context_unification_repo
    )

    rag_indexing_phase = RagIndexingPhase(
        job_manager=job_manager,
        rag_storage_service=rag_storage_service,
        unified_repo=context_unification_repo,
        async_model_service=async_model_service,
        loader=hf_embedder
    )

    # Pipeline Assembly
    return BatchMediaPipeline(
        scan_phase=scan_phase,
        ingest_phase=ingest_phase,
        extract_audio_phase=extract_audio_phase,
        vad_phase=vad_phase,
        domain_adaptation_phase=domain_adaptation_phase,
        transcribe_phase=transcribe_phase,
        hallucination_correction_phase=hallucination_correction_phase,
        diarize_phase=diarize_phase,
        sentence_reconstruction_phase=sentence_reconstruction_phase,
        domain_phonetic_correction_phase=domain_phonetic_correction_phase,
        contextual_retrieval_phase=contextual_retrieval_phase,
        visual_cue_detection_phase=visual_cue_detection_phase,
        frame_extraction_phase=frame_extraction_phase,
        visual_analysis_phase=visual_analysis_phase,
        visual_hallucination_correction_phase=visual_hallucination_correction_phase,
        visual_context_synthesis_phase=visual_context_synthesis_phase,
        context_unification_phase=context_unification_phase,
        clip_extraction_phase=clip_extraction_phase,
        rag_indexing_phase=rag_indexing_phase
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Onyx AI Appliance - Batch Media Pipeline")
    parser.add_argument("target_dir", type=str, help="Absolute or relative path to media files.")
    args = parser.parse_args()

    target_path = Path(args.target_dir).resolve()
    if not target_path.exists() or not target_path.is_dir():
        print(f"[ERROR] Target directory does not exist: {target_path}")
        sys.exit(1)

    print("==================================================")
    print("  ONYX BATCH PIPELINE STARTING")
    print(f"  Target Directory: {target_path}")
    print("==================================================\n")

    bootstrap_database()
    pipeline = get_orchestration_pipeline()

    try:
        results = pipeline.process_directory(str(target_path), SourceType.VIDEO)

        print("\n==================================================")
        print("  PIPELINE EXECUTION COMPLETE")
        print("==================================================")

        for phase_name, batch_result in results.items():
            print(f"\n[{phase_name.value.upper()} PHASE]")
            print(f"  Status:   {batch_result.status.value.upper()}")
            print(f"  Success:  {batch_result.successful_count} items")
            print(f"  Failed:   {batch_result.failed_count} items")
            if batch_result.errors:
                print("  Errors encountered:")
                for err in batch_result.errors:
                    print(f"    - {err}")

        # Post-Pipeline processing and reporting
        storage_repo = StorageRepository(db_manager)

        output_dir = target_path.joinpath("onyx_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        if PipelinePhase.DOMAIN_PHONETIC_CORRECTION in results:
            print("\n==================================================")
            print("  WRITING CORRECTED TRANSCRIPTS TO DISK")
            print("==================================================")

            phonetic_repo = DomainPhoneticRepository(db_manager)
            target_batch = results[PipelinePhase.DOMAIN_PHONETIC_CORRECTION]

            for item in target_batch.items:
                if item.success and item.source_id:
                    source = storage_repo.get_by_id(item.source_id)
                    corrected_transcript = phonetic_repo.get_by_source_id(item.source_id)

                    if source and corrected_transcript:
                        safe_name = Path(source.name).stem
                        out_file = output_dir.joinpath(f"{safe_name}_transcript.txt")

                        try:
                            formatted_text = ""
                            for sentence in corrected_transcript.sentences:
                                speaker = sentence.speaker_id or "Unknown Speaker"
                                formatted_text += f"[{speaker}] {sentence.start_sec:.2f}s - {sentence.end_sec:.2f}s: {sentence.text}\n"

                            with open(out_file, "w", encoding="utf-8") as f:
                                f.write(formatted_text)
                            print(f"  [SUCCESS] Saved Transcript: {out_file}")
                        except IOError as e:
                            print(f"  [ERROR] Failed to write {out_file.name}: {e}")

        if PipelinePhase.VISUAL_ANALYSIS in results:
            print("\n==================================================")
            print("  WRITING DETAILED VISUAL ANALYSIS REPORTS")
            print("==================================================")

            visual_analysis_repo = VisualAnalysisRepository(db_manager)
            contextual_repo = ContextualRepository(db_manager)
            target_batch = results[PipelinePhase.VISUAL_ANALYSIS]

            for item in target_batch.items:
                if item.success and item.source_id:
                    source = storage_repo.get_by_id(item.source_id)
                    transcript = contextual_repo.get_by_source_id(item.source_id)

                    if source and transcript:
                        safe_name = Path(source.name).stem
                        out_file = output_dir.joinpath(f"{safe_name}_visual_report.txt")

                        try:
                            formatted_text = f"Detailed Visual Evidence Analysis for {source.name}\n"
                            formatted_text += "=" * 60 + "\n\n"

                            found_visuals = False
                            for chunk in transcript.chunks:
                                if chunk.needs_visual:
                                    analysis = visual_analysis_repo.get_by_chunk_id(chunk.chunk_id)

                                    if analysis and analysis.analyses:
                                        found_visuals = True
                                        time_str = f"[{chunk.start_sec:.2f}s - {chunk.end_sec:.2f}s]"
                                        formatted_text += f"CHUNK TIMESTAMP: {time_str}\n"
                                        formatted_text += f"CONTEXT TEXT:   {chunk.original_text}\n"

                                        for idx, frame_eval in enumerate(analysis.analyses):
                                            formatted_text += f"  - FRAME {idx + 1} VLM DESCRIPTION: {frame_eval.description}\n"

                                        formatted_text += "-" * 40 + "\n"

                            if not found_visuals:
                                formatted_text += "No relevant visual evidence detected or analyzed in this media file."

                            with open(out_file, "w", encoding="utf-8") as f:
                                f.write(formatted_text)
                            print(f"  [SUCCESS] Saved Detailed Visual Report: {out_file}")
                        except IOError as e:
                            print(f"  [ERROR] Failed to write visual report {out_file.name}: {e}")

        if PipelinePhase.VISUAL_CONTEXT_SYNTHESIS in results:
            print("\n==================================================")
            print("  WRITING FINAL VISUAL CONTEXT NARRATIVE TO DISK")
            print("==================================================")

            visual_synthesis_repo = VisualSynthesisRepository(db_manager)
            target_batch = results[PipelinePhase.VISUAL_CONTEXT_SYNTHESIS]

            for item in target_batch.items:
                if item.success and item.source_id:
                    source = storage_repo.get_by_id(item.source_id)
                    timeline = visual_synthesis_repo.get_by_source_id(item.source_id)

                    if source and timeline:
                        safe_name = Path(source.name).stem
                        out_file = output_dir.joinpath(f"{safe_name}_visual_narrative.txt")

                        try:
                            formatted_text = f"Final Visual Context Narrative for {source.name}\n"
                            formatted_text += "=" * 60 + "\n\n"

                            found_narratives = False
                            for chunk in timeline.visual_chunks:
                                if chunk.visual_context_string:
                                    found_narratives = True
                                    formatted_text += f"Synthesized Narrative: {chunk.visual_context_string}\n"
                                    formatted_text += "-" * 40 + "\n"

                            if not found_narratives:
                                formatted_text += "No final visual narratives synthesized for this media file."

                            with open(out_file, "w", encoding="utf-8") as f:
                                f.write(formatted_text)
                            print(f"  [SUCCESS] Saved Final Visual Narrative: {out_file}")
                        except IOError as e:
                            print(f"  [ERROR] Failed to write visual narrative {out_file.name}: {e}")

        if PipelinePhase.CONTEXT_UNIFICATION in results:
            print("\n==================================================")
            print("  WRITING UNIFIED RAG DOCUMENTS TO DISK")
            print("==================================================")

            context_unification_repo = ContextUnificationRepository(db_manager)
            target_batch = results[PipelinePhase.CONTEXT_UNIFICATION]

            for item in target_batch.items:
                if item.success and item.source_id:
                    source = storage_repo.get_by_id(item.source_id)
                    unified_transcript = context_unification_repo.get_by_source_id(item.source_id)

                    if source and unified_transcript:
                        safe_name = Path(source.name).stem
                        out_file = output_dir.joinpath(f"{safe_name}_unified_rag_document.txt")

                        try:
                            formatted_text = f"Final RAG Unified Document for {source.name}\n"
                            formatted_text += "=" * 60 + "\n\n"

                            for chunk in unified_transcript.chunks:
                                formatted_text += f"[Chunk {chunk.chunk_id}]\n"
                                formatted_text += f"Time: {chunk.start_sec:.2f}s - {chunk.end_sec:.2f}s\n"

                                frame_ids = [str(getattr(evidence, "frame_id", getattr(evidence, "id", "N/A"))) for
                                             evidence in chunk.visual_evidence]
                                formatted_text += f"Frames: {frame_ids}\n"

                                formatted_text += "-" * 40 + "\n"
                                formatted_text += f"{chunk.text_payload}\n\n"

                            with open(out_file, "w", encoding="utf-8") as f:
                                f.write(formatted_text)
                            print(f"  [SUCCESS] Saved Unified Document: {out_file}")
                        except IOError as e:
                            print(f"  [ERROR] Failed to write unified doc {out_file.name}: {e}")

        if PipelinePhase.CLIP_EXTRACTION in results:
            print("\n==================================================")
            print("  CLIP EXTRACTION REPORT")
            print("==================================================")
            target_batch = results[PipelinePhase.CLIP_EXTRACTION]
            for item in target_batch.items:
                if getattr(item, "success", False) and getattr(item, "source_id", None):
                    source = storage_repo.get_by_id(item.source_id)
                    name = source.name if source else str(item.source_id)
                    print(f"  [SUCCESS] {name} -> {item.extracted_clips_count} video clips extracted")
                elif not getattr(item, "success", True):
                    print(f"  [FAILED] {item.source_id}: {getattr(item, 'error_message', 'Unknown error')}")

        # --- Phase 16 Storage Report ---
        if PipelinePhase.RAG_INDEXING in results:
            print("\n==================================================")
            print("  VECTOR DATABASE STORAGE REPORT")
            print("==================================================")

            target_batch = results[PipelinePhase.RAG_INDEXING]
            for item in target_batch.items:
                if item.success and item.source_id:
                    source = storage_repo.get_by_id(item.source_id)
                    if source:
                        print(f"  [SUCCESS] Source: {source.name}")
                        print(f"            - Hybrid RAG Chunks (Vector + FTS): {item.dense_vectors_saved}")

    except Exception as e:
        print(f"\n[FATAL PIPELINE CRASH] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()