# Canvas: test_batch_audio_pipeline_integration_v2
# Path: integration_tests/orchestration/test_batch_audio_pipeline_integration.py

import os
import shutil
import pytest
import json
import gc
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
from app.features.scanner.service.scanner import FileScanner
from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter as AudioFFmpegAdapter
from app.features.audio_extractor.data.repositories import AudioExtractionRepository
from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI

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

from app.features.visual_context_synthesis.data.repositories import VisualSynthesisRepository

from app.features.context_unification.data.repositories import ContextUnificationRepository
from app.features.context_unification.service.unification_api import UnificationAPI

from app.features.rag_storage.data.postgres_rag_repo import PostgresRagRepository
from app.features.rag_storage.data.hf_embedder import HuggingFaceEmbeddingAdapter
from app.features.rag_storage.service.storage_api import RAGStorageService

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
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

from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.pipelines.batch_audio_pipeline import BatchAudioPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus


class TestBatchAudioPipelineIntegration:

    @pytest.fixture(autouse=True)
    def seed_glossary(self):
        settings.GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
        glossary_file = settings.GLOSSARY_DIR / "general.json"

        if not glossary_file.exists():
            mock_data = {
                "base_terms": ["Izvoria", "AI"],
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
                    if obj_type in ["Llama", "WhisperModel", "SentenceTransformer"]:
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

    @pytest.fixture
    def audio_pipeline_setup(self, db_manager, tmp_path):
        job_repo = JobRepository(db_manager)
        job_manager = JobManager(job_repo)

        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        scanner = FileScanner()

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
        domain_adaptation_api = DomainAdaptationAPI(
            repository=domain_repo,
            orchestrator=orchestrator,
            scout_adapter=ScoutWhisperAdapter(),
            llama_adapter=LlamaAdapter(),
            smart_sampler=SmartSampler()
        )

        transcript_repo = TranscriptRepository(db_manager)
        real_transcribe_adapter = FasterWhisperAdapter()
        transcription_api = TranscriptionAPI(transcript_repo, vad_repo, orchestrator, real_transcribe_adapter)
        transcribe_manifest = ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path=settings.WHISPER_MODEL_PATH,
            required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
        )

        correction_api = CorrectionAPI(transcript_repository=transcript_repo, vad_repository=vad_repo)

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

        # Replaced the Mock with the real repository
        visual_synthesis_repo = VisualSynthesisRepository(db_manager)

        context_unification_repo = ContextUnificationRepository(db_manager)
        unification_api = UnificationAPI(repository=context_unification_repo)

        rag_repo = PostgresRagRepository()
        hf_embedder = HuggingFaceEmbeddingAdapter()
        rag_storage_service = RAGStorageService(
            rag_repo=rag_repo,
            orchestrator=orchestrator,
            embedder=hf_embedder
        )

        audio_extractor = AudioExtractorAPI(
            ffmpeg_adapter=AudioFFmpegAdapter(timeout_seconds=300),
            repository=AudioExtractionRepository(db_manager),
            storage_api=storage_api
        )

        pipeline = BatchAudioPipeline(
            scan_phase=ScanPhase(job_manager, scanner),
            ingest_phase=IngestPhase(job_manager, storage_api),
            extract_audio_phase=ExtractAudioPhase(job_manager, audio_extractor),
            vad_phase=VadPhase(job_manager, vad_api, storage_api, async_service, vad_manifest, real_vad_adapter),
            domain_adaptation_phase=DomainAdaptationPhase(job_manager, domain_adaptation_api, audio_extractor, vad_repo),
            transcribe_phase=TranscribePhase(job_manager, transcription_api, audio_extractor, domain_repo, async_service,
                                             transcribe_manifest, real_transcribe_adapter),
            hallucination_correction_phase=HallucinationCorrectionPhase(job_manager, correction_api),
            diarize_phase=DiarizePhase(job_manager, diarization_api, audio_extractor, async_service, diarize_manifest,
                                       real_diarize_adapter),
            sentence_reconstruction_phase=SentenceReconstructionPhase(job_manager, reconstruction_api,
                                                                      transcription_api, async_service,
                                                                      punctuation_manifest, real_punctuation_adapter),
            domain_phonetic_correction_phase=DomainPhoneticCorrectionPhase(job_manager, phonetic_correction_api,
                                                                           sentence_repo, domain_repo, async_service,
                                                                           phonetic_manifest, real_phonetic_adapter),
            contextual_retrieval_phase=ContextualRetrievalPhase(job_manager=job_manager,
                                                                contextualizer_api=contextualizer_api,
                                                                phonetic_repo=phonetic_repo,
                                                                async_model_service=async_service,
                                                                loader=real_slm_adapter),
            context_unification_phase=ContextUnificationPhase(job_manager=job_manager, unification_api=unification_api,
                                                              contextual_repo=contextual_repo,
                                                              visual_synthesis_repo=visual_synthesis_repo),
            rag_indexing_phase=RagIndexingPhase(job_manager=job_manager, rag_storage_service=rag_storage_service,
                                                unified_repo=context_unification_repo,
                                                async_model_service=async_service, loader=hf_embedder)
        )

        yield pipeline
        async_service.shutdown()

    @pytest.mark.integration
    def test_end_to_end_audio_pipeline(self, audio_pipeline_setup, tmp_path):
        pipeline = audio_pipeline_setup

        if not os.path.exists(settings.RAG_EMBEDDING_MODEL_PATH):
            pytest.skip(
                f"Integration test skipped due to missing Embedding model at: {settings.RAG_EMBEDDING_MODEL_PATH}")

        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()

        original_audio = Path("integration_tests/resources/test_speech.wav").resolve()
        if not original_audio.exists():
            pytest.skip(f"Required test file not found at: {original_audio}")

        test_file = source_dir / "test_audio.wav"
        shutil.copy2(original_audio, test_file)

        try:
            results = pipeline.process_directory(str(source_dir))
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping test due to insufficient VRAM: {e}")
            raise e

        # Assert Expected Audio-Only Phases
        expected_phases = [
            PipelinePhase.SCAN, PipelinePhase.INGEST,
            PipelinePhase.VAD, PipelinePhase.DOMAIN_ADAPTATION, PipelinePhase.TRANSCRIBE,
            PipelinePhase.HALLUCINATION_CORRECTION, PipelinePhase.DIARIZE,
            PipelinePhase.SENTENCE_RECONSTRUCTION, PipelinePhase.DOMAIN_PHONETIC_CORRECTION,
            PipelinePhase.CONTEXTUAL_RETRIEVAL,
            PipelinePhase.CONTEXT_UNIFICATION, PipelinePhase.RAG_INDEXING
        ]

        for phase in expected_phases:
            assert phase in results
            assert results[phase].status == PhaseStatus.COMPLETED

        # Audio sources are normalized to 16 kHz mono WAV too
        assert results[PipelinePhase.AUDIO_EXTRACTION].status == PhaseStatus.COMPLETED

        # Ensure Visual/Video phases are missing
        assert PipelinePhase.VISUAL_ANALYSIS not in results
        assert PipelinePhase.CLIP_EXTRACTION not in results