# filename: main_v1.py
import argparse
import sys
from pathlib import Path

# --- Database & Models ---
from app.core.database.service.manager import db_manager
from app.core.database.data.engine import Base
from app.core.jobs.data.models import JobModel
from app.features.storage.data.models import SourceModel, FileModel
from app.features.audio_extractor.data.models import AudioExtractionModel

# --- Core & Feature Imports ---
from app.core.jobs.data.repositories import JobRepository
from app.core.jobs.service.manager import JobManager
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import SourceType
from app.features.scanner.service.scanner import FileScanner
from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter
from app.features.audio_extractor.data.repositories import AudioExtractionRepository
from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI

# --- Orchestration (Service Layer) ---
from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline


def bootstrap_database():
    """Ensures the Postgres database tables exist before running the app."""
    try:
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[FATAL] Could not connect to the database. Ensure Postgres is running.\nError: {e}")
        sys.exit(1)


def get_orchestration_pipeline() -> BatchMediaPipeline:
    """Wires up the dependency injection for the entire application."""
    # 1. Core Core Systems
    job_repo = JobRepository(db_manager)
    job_manager = JobManager(job_repo)

    # 2. Storage System
    storage_repo = StorageRepository(db_manager)
    storage_api = StorageAPI(storage_repo)
    storage_api.file_manager.base_path.mkdir(parents=True, exist_ok=True)  # Ensure physical vault exists

    # 3. Features
    scanner = FileScanner()
    audio_repo = AudioExtractionRepository(db_manager)
    ffmpeg_adapter = FFmpegAdapter()
    audio_extractor = AudioExtractorAPI(
        ffmpeg_adapter=ffmpeg_adapter,
        repository=audio_repo,
        storage_api=storage_api
    )

    # 4. Orchestration Service Layer (Phases)
    scan_phase = ScanPhase(job_manager, scanner)
    ingest_phase = IngestPhase(job_manager, storage_api)
    extract_audio_phase = ExtractAudioPhase(job_manager, audio_extractor)

    # 5. Pipeline Director
    return BatchMediaPipeline(
        scan_phase=scan_phase,
        ingest_phase=ingest_phase,
        extract_audio_phase=extract_audio_phase
    )


def main():
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(description="Onyx AI Appliance - Batch Media Pipeline")
    parser.add_argument(
        "target_dir",
        type=str,
        help="The absolute or relative path to the directory containing media files."
    )
    args = parser.parse_args()

    target_path = Path(args.target_dir).resolve()
    if not target_path.exists() or not target_path.is_dir():
        print(f"[ERROR] Target directory does not exist or is not a directory: {target_path}")
        sys.exit(1)

    print(f"==================================================")
    print(f"  ONYX BATCH PIPELINE STARTING")
    print(f"  Target Directory: {target_path}")
    print(f"==================================================\n")

    # Initialize System
    bootstrap_database()
    pipeline = get_orchestration_pipeline()

    try:
        # Execute the pipeline using the Service Layer
        # Treating everything found as VIDEO so Phase 3 (Audio Extraction) triggers.
        results = pipeline.process_directory(str(target_path), SourceType.VIDEO)

        print(f"\n==================================================")
        print(f"  PIPELINE EXECUTION COMPLETE")
        print(f"==================================================")

        # Display Results
        for phase_name, batch_result in results.items():
            print(f"\n[{phase_name.value.upper()} PHASE]")
            print(f"  Status:   {batch_result.status.value.upper()}")
            print(f"  Success:  {batch_result.successful_count} items")
            print(f"  Failed:   {batch_result.failed_count} items")

            if batch_result.errors:
                print("  Errors encountered:")
                for err in batch_result.errors:
                    print(f"    - {err}")

    except Exception as e:
        print(f"\n[FATAL PIPELINE CRASH] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()