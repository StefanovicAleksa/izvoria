# Canvas: main_v6
# Path: main.py

import argparse
import sys
import json
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

from app.features.diarization.data.repositories import DiarizationRepository
from app.features.diarization.data.nemo_adapter import NeMoDiarizationAdapter
from app.features.diarization.service.diarization_api import DiarizationAPI

# --- NEW IMPORTS FOR INTELLIGENCE BRIDGE ---
from app.features.scene_detector.data.ffmpeg_scene_adapter import FFmpegSceneAdapter
from app.features.scene_detector.utils.phash_deduplicator import PhashDeduplicator
from app.features.scene_detector.service.scene_api import SceneDetectorAPI

from app.features.nlp_router.data.phi3_adapter import Phi3RouterAdapter
from app.features.nlp_router.service.router_api import NLPRouterAPI

from app.features.intelligence_bridge.data.repository import IntelligentSceneRepository
from app.features.intelligence_bridge.service.bridge_api import IntelligenceBridgeAPI
from app.orchestration.service.phases.intelligence_bridge_phase import IntelligenceBridgePhase
# -------------------------------------------

from app.orchestration.domain.enums import PipelinePhase
from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.service.phases.vad_phase import VadPhase
from app.orchestration.service.phases.transcribe_phase import TranscribePhase
from app.orchestration.service.phases.diarize_phase import DiarizePhase
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
        model_path=getattr(settings, "DIARIZATION_MODEL_PATH", "models/titanet_large.nemo"),
        required_vram_mb=getattr(settings, "DIARIZATION_VRAM_REQUIREMENT_MB", 500.0)
    )

    # --- NEW: Intelligence Bridge Setup ---
    scene_adapter = FFmpegSceneAdapter(timeout_seconds=300)
    phash_dedup = PhashDeduplicator(hamming_distance_threshold=5)
    scene_api = SceneDetectorAPI(scene_adapter, storage_api, phash_dedup)

    nlp_manifest = ModelManifest(
        model_type=ModelType.LLM,
        model_path=getattr(settings, "PHI_MODEL_PATH", "models/Phi-3.5-mini-instruct-Q4_K_M.gguf"),
        required_vram_mb=getattr(settings, "NLP_VRAM_REQUIREMENT_MB", 3500.0)
    )
    nlp_adapter = Phi3RouterAdapter()
    nlp_router_api = NLPRouterAPI(model_orchestrator, nlp_adapter, nlp_manifest)

    bridge_repo = IntelligentSceneRepository(db_manager)
    bridge_api = IntelligenceBridgeAPI(bridge_repo, scene_api, transcription_api, nlp_router_api)
    # --------------------------------------

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

    diarize_phase = DiarizePhase(
        job_manager=job_manager,
        diarization_api=diarization_api,
        storage_api=storage_api,
        async_model_service=async_model_service,
        diarize_manifest=diarize_manifest,
        diarize_adapter=diarize_adapter
    )

    bridge_phase = IntelligenceBridgePhase(
        job_manager=job_manager,
        bridge_api=bridge_api,
        storage_api=storage_api,
        async_model_service=async_model_service,
        nlp_manifest=nlp_manifest,
        nlp_adapter=nlp_adapter
    )

    return BatchMediaPipeline(
        scan_phase=scan_phase,
        ingest_phase=ingest_phase,
        extract_audio_phase=extract_audio_phase,
        vad_phase=vad_phase,
        transcribe_phase=transcribe_phase,
        diarize_phase=diarize_phase,
        intelligence_bridge_phase=bridge_phase
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

        storage_repo = StorageRepository(db_manager)

        # --- Output Transcripts to .txt files ---
        if PipelinePhase.DIARIZE in results or PipelinePhase.TRANSCRIBE in results:
            print("\n==================================================")
            print("  WRITING TRANSCRIPTS TO DISK")
            print("==================================================")

            transcript_repo = TranscriptRepository(db_manager)
            target_batch = results.get(PipelinePhase.DIARIZE) or results.get(PipelinePhase.TRANSCRIBE)

            for item in target_batch.items:
                if item.success and item.source_id:
                    source = storage_repo.get_by_id(item.source_id)
                    transcript = transcript_repo.get_by_source_id(item.source_id)

                    if source and transcript:
                        safe_name = Path(source.name).stem
                        out_file = target_path / f"{safe_name}_transcript.txt"

                        try:
                            formatted_text = ""
                            for seg in transcript.segments:
                                speaker = seg.speaker_id or "Unknown Speaker"
                                formatted_text += f"[{speaker}] {seg.start_sec:.2f}s - {seg.end_sec:.2f}s: {seg.text}\n"

                            with open(out_file, "w", encoding="utf-8") as f:
                                f.write(formatted_text)
                            print(f"  [SUCCESS] Saved: {out_file.name}")
                        except IOError as e:
                            print(f"  [ERROR] Failed to write {out_file.name}: {e}")

        # --- Output Scene Maps to .json files ---
        if PipelinePhase.INTELLIGENCE_BRIDGE in results:
            print("\n==================================================")
            print("  WRITING INTELLIGENCE SCENE MAPS TO DISK")
            print("==================================================")

            bridge_repo = IntelligentSceneRepository(db_manager)
            target_batch = results[PipelinePhase.INTELLIGENCE_BRIDGE]

            for item in target_batch.items:
                if item.success and item.source_id:
                    source = storage_repo.get_by_id(item.source_id)
                    scene_map = bridge_repo.get_map_by_source_id(item.source_id)

                    if source and scene_map:
                        safe_name = Path(source.name).stem
                        out_file = target_path / f"{safe_name}_scenes.json"

                        try:
                            # Dump the Pydantic models to formatted JSON
                            map_dict = scene_map.model_dump(mode='json')
                            with open(out_file, "w", encoding="utf-8") as f:
                                json.dump(map_dict, f, indent=4)
                            print(f"  [SUCCESS] Saved: {out_file.name}")
                        except IOError as e:
                            print(f"  [ERROR] Failed to write {out_file.name}: {e}")

    except Exception as e:
        print(f"\n[FATAL PIPELINE CRASH] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()