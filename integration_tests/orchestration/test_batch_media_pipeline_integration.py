# Canvas: test_batch_media_pipeline_integration_v37
# Path: integration_tests/orchestration/test_batch_media_pipeline_integration.py

import subprocess
import pytest
import shutil
import os
import gc
import json
import typing
from pathlib import Path

from app.core.config import settings
from app.core.jobs.service.manager import JobManager
from app.core.jobs.data.repositories import JobRepository
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType
from app.features.scanner.service.scanner import FileScanner

from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI
from app.features.audio_extractor.data.repositories import AudioExtractionRepository
from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter as AudioFFmpegAdapter

from app.features.vad.data.repositories import VadRepository
from app.features.vad.service.vad_api import VadAPI
from app.features.vad.data.nemo_adapter import NeMoVadAdapter

from app.features.domain_adaptation.data.repositories import DomainAdaptationRepository
from app.features.domain_adaptation.data.scout_adapter import ScoutWhisperAdapter
from app.features.domain_adaptation.data.llama_adapter import LlamaAdapter
from app.features.domain_adaptation.utils.smart_sampler import SmartSampler
from app.features.domain_adaptation.service.adaptation_api import DomainAdaptationAPI

from app.features.transcription.data.repositories import TranscriptRepository
from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter

from app.features.hallucination_correction.service.correction_api import CorrectionAPI

from app.features.diarization.data.repositories import DiarizationRepository
from app.features.diarization.service.diarization_api import DiarizationAPI
from app.features.diarization.data.nemo_adapter import NeMoDiarizationAdapter

from app.features.sentence_reconstruction.data.repositories import PostgresSentenceRepository
from app.features.sentence_reconstruction.service.reconstruction_api import SentenceReconstructionAPI
from app.features.sentence_reconstruction.data.punctuation_adapter import PunctuationAdapter

from app.features.domain_phonetic_correction.data.repositories import DomainPhoneticRepository
from app.features.domain_phonetic_correction.data.qwen_adapter import QwenDomainAdapter
from app.features.domain_phonetic_correction.service.correction_api import DomainPhoneticCorrectionAPI

from app.features.contextual_retrieval.data.repositories import ContextualRepository
from app.features.contextual_retrieval.data.slm_adapter import SLMContextAdapter
from app.features.contextual_retrieval.service.contextualizer_api import ContextualizerAPI

from app.features.visual_cue_detection.data.qwen_detector_adapter import QwenDetectorAdapter
from app.features.visual_cue_detection.service.detection_api import VisualCueDetectionAPI

from app.features.frame_extraction.data.ffmpeg_adapter import FFmpegExtractionAdapter as FrameFFmpegAdapter
from app.features.frame_extraction.data.repositories import FrameExtractionRepository
from app.features.frame_extraction.service.extraction_api import FrameExtractionAPI
from app.features.frame_extraction.utils.quality_filter import QualityFilter
from app.features.frame_extraction.utils.frame_deduplicator import FrameDeduplicator

from app.features.visual_analysis.data.vlm_adapter import QwenVLMAdapter
from app.features.visual_analysis.data.repositories import VisualAnalysisRepository
from app.features.visual_analysis.service.analysis_api import VisualAnalysisAPI

from app.features.visual_hallucination_correction.data.slm_adapter import QwenHallucinationAdapter
from app.features.visual_hallucination_correction.data.repositories import VisualHallucinationRepository
from app.features.visual_hallucination_correction.service.correction_api import VisualHallucinationCorrectionAPI

from app.features.visual_context_synthesis.data.slm_adapter import VisualSLMAdapter
from app.features.visual_context_synthesis.data.repositories import VisualSynthesisRepository
from app.features.visual_context_synthesis.service.synthesis_api import VisualSynthesisAPI

from app.features.context_unification.data.repositories import ContextUnificationRepository
from app.features.context_unification.service.unification_api import UnificationAPI

from app.features.clip_extractor.data.ffmpeg_video_adapter import FFmpegVideoAdapter as ClipFFmpegAdapter
from app.features.clip_extractor.data.repositories import VideoClipRepository
from app.features.clip_extractor.service.extractor_api import ClipExtractorAPI

from app.features.rag_storage.data.postgres_rag_repo import PostgresRagRepository
from app.features.rag_storage.data.hf_embedder import HuggingFaceEmbeddingAdapter
from app.features.rag_storage.service.storage_api import RAGStorageService

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
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus


class TestBatchMediaPipelineIntegration:

    @pytest.fixture(autouse=True)
    def seed_glossary(self):
        """Seeds a mock glossary to ensure Phase 5 (Domain Adaptation) passes."""
        settings.GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
        glossary_file = settings.GLOSSARY_DIR / "general.json"

        if not glossary_file.exists():
            mock_data = {
                "base_terms": ["Onyx", "AI"],
                "sub_domains": {
                    "legal": {"anchor_words": ["affidavit", "plaintiff"], "terms": ["litigation"]},
                    "medical": {"anchor_words": ["diagnosis", "cardiology"], "terms": ["prognosis"]}
                }
            }
            with open(glossary_file, 'w') as f:
                json.dump(mock_data, f)

    @pytest.fixture(autouse=True, scope="function")
    def vram_cleanup(self):
        def clear_gpu():
            gc.collect()
            for obj in gc.get_objects():
                try:
                    obj_type = type(obj).__name__
                    if obj_type in ["Llama", "WhisperModel", "Qwen2VLForConditionalGeneration", "SentenceTransformer"]:
                        if hasattr(obj, "close"):
                            obj.close()
                except Exception:
                    pass
            gc.collect()
            try:
                import torch
            except ImportError:
                torch = None

            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        clear_gpu()
        yield
        clear_gpu()

    @pytest.fixture(autouse=True)
    def check_ffmpeg_installed(self):
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except FileNotFoundError:
            pytest.skip("FFmpeg is not installed on this system.")

    @pytest.fixture
    def pipeline_setup(self, db_manager, tmp_path):
        job_repo = JobRepository(db_manager)
        job_manager = JobManager(job_repo)

        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        scanner = FileScanner()

        audio_repo = AudioExtractionRepository(db_manager)
        ffmpeg_audio_adapter = AudioFFmpegAdapter(timeout_seconds=300)
        audio_extractor = AudioExtractorAPI(
            ffmpeg_adapter=ffmpeg_audio_adapter,
            repository=audio_repo,
            storage_api=storage_api
        )

        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)
        bg_worker = BackgroundModelWorker(orchestrator)
        async_service = AsyncModelService(bg_worker)

        vad_repo = VadRepository(db_manager)
        real_vad_adapter = NeMoVadAdapter()
        vad_api = VadAPI(vad_repo, orchestrator, real_vad_adapter)
        vad_manifest = ModelManifest(
            model_type=ModelType.VAD,
            model_path=settings.VAD_MODEL_PATH,
            required_vram_mb=settings.VAD_VRAM_REQUIREMENT_MB
        )

        domain_repo = DomainAdaptationRepository(db_manager)
        real_scout_adapter = ScoutWhisperAdapter()
        real_classifier_adapter = LlamaAdapter()
        sampler = SmartSampler()
        domain_adaptation_api = DomainAdaptationAPI(
            repository=domain_repo,
            orchestrator=orchestrator,
            scout_adapter=real_scout_adapter,
            llama_adapter=real_classifier_adapter,
            smart_sampler=sampler
        )

        transcript_repo = TranscriptRepository(db_manager)
        real_transcribe_adapter = FasterWhisperAdapter()
        transcription_api = TranscriptionAPI(transcript_repo, vad_repo, orchestrator, real_transcribe_adapter)
        transcribe_manifest = ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path=settings.WHISPER_MODEL_PATH,
            required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
        )

        correction_api = CorrectionAPI(
            transcript_repository=transcript_repo,
            vad_repository=vad_repo
        )

        diarize_repo = DiarizationRepository(db_manager)
        real_diarize_adapter = NeMoDiarizationAdapter()
        diarization_api = DiarizationAPI(diarize_repo, vad_repo, transcription_api, storage_api, orchestrator,
                                         real_diarize_adapter)
        diarize_manifest = ModelManifest(
            model_type=ModelType.DIARIZATION,
            model_path=settings.DIARIZATION_MODEL_PATH,
            required_vram_mb=settings.DIARIZATION_VRAM_REQUIREMENT_MB
        )

        sentence_repo = PostgresSentenceRepository(db_manager)
        real_punctuation_adapter = PunctuationAdapter()
        reconstruction_api = SentenceReconstructionAPI(sentence_repo, orchestrator, real_punctuation_adapter)
        punctuation_path = getattr(settings, "PUNCTUATION_MODEL_PATH", "models/fullstop-punctuation-multilingual-base")
        punctuation_manifest = ModelManifest(
            model_type=ModelType.SENTENCE_RECONSTRUCTION,
            model_path=punctuation_path,
            required_vram_mb=getattr(settings, "PUNCTUATION_VRAM_REQUIREMENT_MB", 500.0)
        )

        phonetic_repo = DomainPhoneticRepository(db_manager)
        real_phonetic_adapter = QwenDomainAdapter()
        phonetic_correction_api = DomainPhoneticCorrectionAPI(
            repository=phonetic_repo,
            orchestrator=orchestrator,
            patch_adapter=real_phonetic_adapter
        )
        phonetic_manifest = ModelManifest(
            model_type=ModelType.DOMAIN_PHONETIC_CORRECTION,
            model_path=settings.DOMAIN_PHONETIC_CORRECTION_MODEL_PATH,
            required_vram_mb=settings.DOMAIN_PHONETIC_CORRECTION_VRAM_REQUIREMENT_MB
        )

        contextual_repo = ContextualRepository(db_manager)
        real_slm_adapter = SLMContextAdapter()
        contextualizer_api = ContextualizerAPI(
            repository=contextual_repo,
            orchestrator=orchestrator,
            slm_adapter=real_slm_adapter
        )

        cue_detector_adapter = QwenDetectorAdapter()
        visual_cue_api = VisualCueDetectionAPI(
            orchestrator=orchestrator,
            adapter=cue_detector_adapter,
            repository=contextual_repo
        )

        frame_repo = FrameExtractionRepository(db_manager)
        ffmpeg_ext_adapter = FrameFFmpegAdapter()
        quality_filter = QualityFilter()
        deduplicator = FrameDeduplicator()

        # FIX: We use 'typing.cast' + PyCharm suppression to satisfy strict type checkers
        # for our structural typing (duck typing) implementation.
        # noinspection PyTypeChecker
        frame_extraction_api = FrameExtractionAPI(
            repository=frame_repo,
            ffmpeg_adapter=typing.cast('typing.Any', ffmpeg_ext_adapter),  # type: ignore
            storage_api=storage_api,
            quality_filter=quality_filter,
            deduplicator=deduplicator
        )

        visual_analysis_repo = VisualAnalysisRepository(db_manager)
        vlm_adapter = QwenVLMAdapter()
        visual_analysis_api = VisualAnalysisAPI(
            repository=visual_analysis_repo,
            vlm_adapter=vlm_adapter,
            storage_api=storage_api
        )

        visual_hallucination_repo = VisualHallucinationRepository(db_manager)
        visual_hallucination_adapter = QwenHallucinationAdapter()
        visual_hallucination_api = VisualHallucinationCorrectionAPI(
            repository=visual_hallucination_repo,
            adapter=visual_hallucination_adapter,
            contextual_repo=contextual_repo,
            visual_analysis_repo=visual_analysis_repo,
            orchestrator=orchestrator
        )

        visual_synthesis_repo = VisualSynthesisRepository(db_manager)
        visual_synthesis_adapter = VisualSLMAdapter()
        visual_synthesis_api = VisualSynthesisAPI(
            repository=visual_synthesis_repo,
            orchestrator=orchestrator,
            slm_adapter=visual_synthesis_adapter,
            visual_analysis_repo=visual_analysis_repo,
            hallucination_repo=visual_hallucination_repo
        )

        context_unification_repo = ContextUnificationRepository(db_manager)
        unification_api = UnificationAPI(repository=context_unification_repo)

        clip_repo = VideoClipRepository(db_manager)
        ffmpeg_video_adapter = ClipFFmpegAdapter(timeout_seconds=60)
        clip_extractor_api = ClipExtractorAPI(
            ffmpeg_adapter=ffmpeg_video_adapter,
            repository=clip_repo,
            storage_api=storage_api
        )

        rag_repo = PostgresRagRepository()
        hf_embedder = HuggingFaceEmbeddingAdapter()
        rag_storage_service = RAGStorageService(
            rag_repo=rag_repo,
            orchestrator=orchestrator,
            embedder=hf_embedder
        )

        scan_phase = ScanPhase(job_manager, scanner)
        ingest_phase = IngestPhase(job_manager, storage_api)
        extract_audio_phase = ExtractAudioPhase(job_manager, audio_extractor)
        vad_phase = VadPhase(job_manager, vad_api, storage_api, async_service, vad_manifest, real_vad_adapter)
        domain_adaptation_phase = DomainAdaptationPhase(job_manager, domain_adaptation_api, audio_extractor, vad_repo)
        transcribe_phase = TranscribePhase(job_manager, transcription_api, audio_extractor, domain_repo, async_service,
                                           transcribe_manifest, real_transcribe_adapter)
        hallucination_correction_phase = HallucinationCorrectionPhase(job_manager, correction_api)
        diarize_phase = DiarizePhase(job_manager, diarization_api, audio_extractor, async_service, diarize_manifest,
                                     real_diarize_adapter)
        sentence_phase = SentenceReconstructionPhase(job_manager, reconstruction_api, transcription_api, async_service,
                                                     punctuation_manifest, real_punctuation_adapter)
        domain_phonetic_phase = DomainPhoneticCorrectionPhase(job_manager, phonetic_correction_api, sentence_repo,
                                                              domain_repo, async_service, phonetic_manifest,
                                                              real_phonetic_adapter)
        contextual_phase = ContextualRetrievalPhase(job_manager=job_manager, contextualizer_api=contextualizer_api,
                                                    phonetic_repo=phonetic_repo, async_model_service=async_service,
                                                    loader=real_slm_adapter)
        visual_cue_phase = VisualCueDetectionPhase(job_manager=job_manager, detection_api=visual_cue_api,
                                                   contextual_repo=contextual_repo, async_model_service=async_service,
                                                   loader=cue_detector_adapter)
        frame_extraction_phase = FrameExtractionPhase(job_manager, frame_extraction_api, contextual_repo, storage_api)
        visual_analysis_phase = VisualAnalysisPhase(job_manager=job_manager, analysis_api=visual_analysis_api,
                                                    contextual_repo=contextual_repo, frame_repo=frame_repo,
                                                    async_model_service=async_service, loader=vlm_adapter)
        visual_hallucination_correction_phase = VisualHallucinationCorrectionPhase(job_manager=job_manager,
                                                                                   correction_api=visual_hallucination_api,
                                                                                   async_model_service=async_service,
                                                                                   loader=visual_hallucination_adapter)

        visual_context_synthesis_phase = VisualContextSynthesisPhase(
            job_manager=job_manager,
            synthesis_api=visual_synthesis_api,
            contextual_repo=contextual_repo,
            async_model_service=async_service,
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
            async_model_service=async_service,
            loader=hf_embedder
        )

        pipeline = BatchMediaPipeline(
            scan_phase=scan_phase,
            ingest_phase=ingest_phase,
            extract_audio_phase=extract_audio_phase,
            vad_phase=vad_phase,
            domain_adaptation_phase=domain_adaptation_phase,
            transcribe_phase=transcribe_phase,
            hallucination_correction_phase=hallucination_correction_phase,
            diarize_phase=diarize_phase,
            sentence_reconstruction_phase=sentence_phase,
            domain_phonetic_correction_phase=domain_phonetic_phase,
            contextual_retrieval_phase=contextual_phase,
            visual_cue_detection_phase=visual_cue_phase,
            frame_extraction_phase=frame_extraction_phase,
            visual_analysis_phase=visual_analysis_phase,
            visual_hallucination_correction_phase=visual_hallucination_correction_phase,
            visual_context_synthesis_phase=visual_context_synthesis_phase,
            context_unification_phase=context_unification_phase,
            clip_extraction_phase=clip_extraction_phase,
            rag_indexing_phase=rag_indexing_phase
        )

        yield pipeline, storage_api
        async_service.shutdown()

    @pytest.mark.integration
    def test_end_to_end_media_pipeline(self, pipeline_setup, tmp_path):
        pipeline, storage_api = pipeline_setup

        if not os.path.exists(settings.RAG_EMBEDDING_MODEL_PATH):
            pytest.skip(
                f"Integration test skipped due to missing Phase 16 Embedding model at: {settings.RAG_EMBEDDING_MODEL_PATH}")

        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()

        original_vid = Path("integration_tests/resources/test_video.mp4").resolve()
        if not original_vid.exists():
            pytest.skip(f"Required test file not found at: {original_vid}")

        test_file = source_dir / "test_video.mp4"
        shutil.copy2(original_vid, test_file)

        try:
            results = pipeline.process_directory(str(source_dir), SourceType.VIDEO)
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping test due to insufficient VRAM: {e}")
            raise e

        # Assert Pipeline Presence
        expected_phases = [
            PipelinePhase.SCAN, PipelinePhase.INGEST, PipelinePhase.AUDIO_EXTRACTION,
            PipelinePhase.VAD, PipelinePhase.DOMAIN_ADAPTATION, PipelinePhase.TRANSCRIBE,
            PipelinePhase.HALLUCINATION_CORRECTION, PipelinePhase.DIARIZE,
            PipelinePhase.SENTENCE_RECONSTRUCTION, PipelinePhase.DOMAIN_PHONETIC_CORRECTION,
            PipelinePhase.CONTEXTUAL_RETRIEVAL, PipelinePhase.VISUAL_CUE_DETECTION,
            PipelinePhase.FRAME_EXTRACTION, PipelinePhase.VISUAL_ANALYSIS,
            PipelinePhase.VISUAL_HALLUCINATION_CORRECTION, PipelinePhase.VISUAL_CONTEXT_SYNTHESIS,
            PipelinePhase.CONTEXT_UNIFICATION, PipelinePhase.CLIP_EXTRACTION,
            PipelinePhase.RAG_INDEXING
        ]

        for phase in expected_phases:
            assert phase in results
            assert results[phase].status == PhaseStatus.COMPLETED