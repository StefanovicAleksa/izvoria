# Canvas: main_v4
# Path: main.py
import argparse
import sys
from pathlib import Path

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

from app.features.transcription.data.repositories import TranscriptRepository
from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
from app.features.transcription.service.transcription_api import TranscriptionAPI

from app.orchestration.domain.enums import PipelinePhase
from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.phases.vad_phase import VadPhase
from app.orchestration.service.phases.transcribe_phase import TranscribePhase
from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline


def bootstrap_database():
    try:
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[FATAL] Could not connect to the database. Ensure Postgres is running.\nError: {e}")
        sys.exit(1)


def get_orchestration_pipeline() -> BatchMediaPipeline:
    job_repo = JobRepository(db_manager)
    job_manager = JobManager(job_repo)
    vram_monitor = PyTorchVramMonitor()
    model_orchestrator = ModelManagerOrchestrator(vram_monitor)

    bg_worker = BackgroundModelWorker(model_orchestrator)
    async_model_service = AsyncModelService(bg_worker)

    storage_repo = StorageRepository(db_manager)
    storage_api = StorageAPI(storage_repo)
    storage_api.file_manager.base_path.mkdir(parents=True, exist_ok=True)

    scanner = FileScanner()
    audio_repo = AudioExtractionRepository(db_manager)
    ffmpeg_adapter = FFmpegAdapter()
    audio_extractor = AudioExtractorAPI(ffmpeg_adapter, audio_repo, storage_api)

    vad_repo = VadRepository(db_manager)
    nemo_adapter = NeMoVadAdapter()
    vad_api = VadAPI(vad_repo, model_orchestrator, nemo_adapter)

    vad_manifest = ModelManifest(
        model_type=ModelType.VAD,
        model_path=settings.VAD_MODEL_PATH,
        required_vram_mb=settings.VAD_VRAM_REQUIREMENT_MB
    )

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

    # Pipeline Phases
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

    transcribe_phase = TranscribePhase(
        job_manager=job_manager,
        transcription_api=transcription_api,
        storage_api=storage_api,
        async_model_service=async_model_service,
        transcribe_manifest=transcribe_manifest,
        transcribe_adapter=whisper_adapter
    )

    return BatchMediaPipeline(
        scan_phase=scan_phase,
        ingest_phase=ingest_phase,
        extract_audio_phase=extract_audio_phase,
        vad_phase=vad_phase,
        transcribe_phase=transcribe_phase
    )


def main():
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

        # --- NEW: Output Transcripts to .txt files ---
        if PipelinePhase.TRANSCRIBE in results:
            print("\n==================================================")
            print("  WRITING TRANSCRIPTS TO DISK")
            print("==================================================")

            # Instantiate repositories to fetch data securely from DB
            storage_repo = StorageRepository(db_manager)
            transcript_repo = TranscriptRepository(db_manager)

            transcribe_batch = results[PipelinePhase.TRANSCRIBE]
            for item in transcribe_batch.items:
                if item.success and item.source_id:
                    source = storage_repo.get_by_id(item.source_id)
                    transcript = transcript_repo.get_by_source_id(item.source_id)

                    if source and transcript:
                        # Strip extension from original video and append .txt
                        safe_name = Path(source.name).stem
                        out_file = target_path / f"{safe_name}_transcript.txt"

                        try:
                            with open(out_file, "w", encoding="utf-8") as f:
                                f.write(transcript.full_text)
                            print(f"  [SUCCESS] Saved: {out_file.name}")
                        except IOError as e:
                            print(f"  [ERROR] Failed to write {out_file.name}: {e}")

    except Exception as e:
        print(f"\n[FATAL PIPELINE CRASH] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()