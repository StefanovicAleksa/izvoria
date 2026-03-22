# Canvas: main_v16
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

from app.features.diarization.data.repositories import DiarizationRepository
from app.features.diarization.data.nemo_adapter import NeMoDiarizationAdapter
from app.features.diarization.service.diarization_api import DiarizationAPI

from app.features.sentence_reconstruction.data.repositories import PostgresSentenceRepository
from app.features.sentence_reconstruction.data.punctuation_adapter import PunctuationAdapter
from app.features.sentence_reconstruction.service.reconstruction_api import SentenceReconstructionAPI

from app.features.phonetic_correction.data.repositories import PhoneticRepository
from app.features.phonetic_correction.data.qwen_adapter import QwenAdapter
from app.features.phonetic_correction.service.correction_api import PhoneticCorrectionAPI

from app.orchestration.domain.enums import PipelinePhase
from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.phases.vad_phase import VadPhase
from app.orchestration.service.phases.domain_adaptation_phase import DomainAdaptationPhase
from app.orchestration.service.phases.transcribe_phase import TranscribePhase
from app.orchestration.service.phases.diarize_phase import DiarizePhase
from app.orchestration.service.phases.sentence_reconstruction_phase import SentenceReconstructionPhase
from app.orchestration.service.phases.phonetic_correction_phase import PhoneticCorrectionPhase
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

    # CRITICAL FIX: Updated Sampler initialization to match new Trailer approach
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

    # 7. Sentence Reconstruction (Phase 7)
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

    # 8. Phonetic Correction (Phase 8)
    phonetic_repo = PhoneticRepository(db_manager)
    phonetic_adapter = QwenAdapter()
    phonetic_correction_api = PhoneticCorrectionAPI(
        repository=phonetic_repo,
        orchestrator=model_orchestrator,
        patch_adapter=phonetic_adapter
    )

    phonetic_manifest = ModelManifest(
        model_type=ModelType.ASR_CORRECTION_PHASE_3,
        model_path=settings.ASR_PHASE_3_MODEL_PATH,
        required_vram_mb=settings.ASR_PHASE_3_VRAM_REQUIREMENT_MB
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

    phonetic_correction_phase = PhoneticCorrectionPhase(
        job_manager=job_manager,
        correction_api=phonetic_correction_api,
        sentence_repo=sentence_repo,
        domain_repo=domain_repo,
        async_model_service=async_model_service,
        manifest=phonetic_manifest,
        adapter=phonetic_adapter
    )

    # Pipeline Assembly
    return BatchMediaPipeline(
        scan_phase=scan_phase,
        ingest_phase=ingest_phase,
        extract_audio_phase=extract_audio_phase,
        vad_phase=vad_phase,
        domain_adaptation_phase=domain_adaptation_phase,
        transcribe_phase=transcribe_phase,
        diarize_phase=diarize_phase,
        sentence_reconstruction_phase=sentence_reconstruction_phase,
        phonetic_correction_phase=phonetic_correction_phase
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

        storage_repo = StorageRepository(db_manager)

        if PipelinePhase.PHONETIC_CORRECTION in results:
            print("\n==================================================")
            print("  WRITING CORRECTED TRANSCRIPTS TO DISK")
            print("==================================================")

            phonetic_repo = PhoneticRepository(db_manager)
            target_batch = results[PipelinePhase.PHONETIC_CORRECTION]

            output_dir = target_path / "output_transcripts"
            output_dir.mkdir(parents=True, exist_ok=True)

            for item in target_batch.items:
                if item.success and item.source_id:
                    source = storage_repo.get_by_id(item.source_id)
                    corrected_transcript = phonetic_repo.get_by_source_id(item.source_id)

                    if source and corrected_transcript:
                        safe_name = Path(source.name).stem
                        out_file = output_dir / f"{safe_name}_transcript.txt"

                        try:
                            formatted_text = ""
                            for sentence in corrected_transcript.sentences:
                                speaker = sentence.speaker_id or "Unknown Speaker"
                                formatted_text += f"[{speaker}] {sentence.start_sec:.2f}s - {sentence.end_sec:.2f}s: {sentence.text}\n"

                            with open(out_file, "w", encoding="utf-8") as f:
                                f.write(formatted_text)
                            print(f"  [SUCCESS] Saved: {out_file}")
                        except IOError as e:
                            print(f"  [ERROR] Failed to write {out_file.name}: {e}")

    except Exception as e:
        print(f"\n[FATAL PIPELINE CRASH] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()