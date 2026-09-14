# Canvas: onyx_engine_v4
# Path: app/engine/service/onyx_engine.py

import logging
import threading
import time
import sys
import os
import uuid
import gc
from typing import Dict

from app.core.database.service.manager import db_manager
from app.core.database.data.engine import Base
from app.core.model_manager.service.async_model_service import AsyncModelService

from app.engine.domain.interfaces import IOnyxEngine
from app.engine.domain.enums import EngineState
from app.engine.domain.entities import IngestionReport
from app.engine.exceptions.lifecycle import EngineOfflineError, EngineBusyError
from app.engine.utils.error_mapper import map_engine_error

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import SourceType
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult
from app.features.qa_generation.domain.entities import QAResponse

from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline
from app.orchestration.service.pipelines.batch_audio_pipeline import BatchAudioPipeline
from app.orchestration.service.pipelines.live_retrieval_pipeline import LiveRetrievalPipeline
from app.orchestration.service.pipelines.live_verified_retrieval_pipeline import LiveVerifiedRetrievalPipeline
from app.orchestration.service.pipelines.live_qa_pipeline import LiveQAPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult

logger = logging.getLogger(__name__)


class OnyxEngine(IOnyxEngine):
    """
    The unified Facade for the Onyx AI Appliance.
    Strictly coordinates State Locks and isolates pipelines.
    """

    def __init__(
            self,
            async_model_service: AsyncModelService,
            media_pipeline: BatchMediaPipeline,
            audio_pipeline: BatchAudioPipeline,
            live_pipeline: LiveRetrievalPipeline,
            live_verified_pipeline: LiveVerifiedRetrievalPipeline,
            live_qa_pipeline: LiveQAPipeline,
            storage_api: StorageAPI
    ):
        self._async_service = async_model_service
        self._media_pipeline = media_pipeline
        self._audio_pipeline = audio_pipeline
        self._live_pipeline = live_pipeline
        self._live_verified_pipeline = live_verified_pipeline
        self._live_qa_pipeline = live_qa_pipeline
        self._storage_api = storage_api

        self._state = EngineState.OFFLINE
        self._state_lock = threading.RLock()

    # ==========================================
    # LIFECYCLE MANAGEMENT
    # ==========================================

    def start(self) -> None:
        with self._state_lock:
            if self._state != EngineState.OFFLINE:
                return

            self._state = EngineState.BOOTING
            logger.info("Booting Onyx Enterprise AI Engine...")

            try:
                with db_manager.session() as session:
                    engine = session.get_bind()
                    Base.metadata.create_all(bind=engine)

                self._force_aggressive_garbage_collection()

                self._state = EngineState.ONLINE
                logger.info("Onyx Engine is ONLINE and ready for tasks.")

            except Exception as e:
                self._state = EngineState.ERROR
                raise map_engine_error(e, context="Starting Engine")

    def stop(self) -> None:
        with self._state_lock:
            if self._state == EngineState.OFFLINE:
                return

            logger.info("Shutting down Onyx Engine...")
            try:
                self._force_aggressive_garbage_collection()

                self._async_service.shutdown()
                db_manager.dispose()

                self._state = EngineState.OFFLINE
                logger.info("Onyx Engine is securely OFFLINE.")
            except Exception as e:
                self._state = EngineState.ERROR
                raise map_engine_error(e, context="Stopping Engine")

    def get_state(self) -> EngineState:
        with self._state_lock:
            return self._state

    # ==========================================
    # VRAM ORCHESTRATION GUARDS
    # ==========================================

    def _prepare_for_ingestion(self) -> None:
        """Locks the engine for batch processing."""
        with self._state_lock:
            if self._state == EngineState.OFFLINE:
                raise EngineOfflineError("Cannot ingest media: Engine is offline.")
            if self._state == EngineState.INGESTING:
                raise EngineBusyError("Cannot ingest media: Hardware is currently locked by another ingestion process.")
            self._state = EngineState.INGESTING

        logger.info("Locking Engine for batch ingestion...")
        try:
            self._force_aggressive_garbage_collection()
        except Exception as e:
            with self._state_lock:
                self._state = EngineState.ONLINE
            raise map_engine_error(e, context="VRAM Eviction for Ingestion")

    def _cleanup_after_ingestion(self) -> None:
        """Unlocks the engine and clears any lingering pipeline artifacts."""
        try:
            logger.info("Batch ingestion finished. Purging pipeline artifacts from VRAM...")
            self._async_service.release_all()
            self._force_aggressive_garbage_collection()
        except Exception as e:
            logger.error(f"Failed to cleanly purge VRAM: {e}")
        finally:
            with self._state_lock:
                self._state = EngineState.ONLINE

    @staticmethod
    def _force_aggressive_garbage_collection() -> None:
        """Forces the destruction of stale memory objects holding heavy weights."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    # ==========================================
    # BATCH PIPELINE DELEGATION
    # ==========================================

    def ingest_videos(self, target_path: str) -> IngestionReport:
        self._prepare_for_ingestion()
        start_time = time.time()
        errors: Dict[str, str] = {}
        metrics: Dict[str, int] = {"scanned": 0, "success": 0, "failed": 0, "vectors": 0, "clips": 0}

        try:
            results = self._media_pipeline.process_directory(target_path, SourceType.VIDEO)
            self._aggregate_metrics(results, errors, metrics)
        except Exception as e:
            raise map_engine_error(e, context=f"Video Ingestion [{target_path}]")
        finally:
            self._cleanup_after_ingestion()

        return self._build_report(target_path, metrics, errors, time.time() - start_time)

    def ingest_audios(self, target_path: str) -> IngestionReport:
        self._prepare_for_ingestion()
        start_time = time.time()
        errors: Dict[str, str] = {}
        metrics: Dict[str, int] = {"scanned": 0, "success": 0, "failed": 0, "vectors": 0, "clips": 0}

        try:
            results = self._audio_pipeline.process_directory(target_path)
            self._aggregate_metrics(results, errors, metrics)
        except Exception as e:
            raise map_engine_error(e, context=f"Audio Ingestion [{target_path}]")
        finally:
            self._cleanup_after_ingestion()

        return self._build_report(target_path, metrics, errors, time.time() - start_time)

    # ==========================================
    # LIVE INFERENCE DELEGATION
    # ==========================================

    def search_standard(self, query: SearchQuery) -> SearchResult:
        self._enforce_online_state()
        try:
            search_id = uuid.uuid4()
            result = self._live_pipeline.execute_search(search_id, query)
            self._resolve_physical_paths(result)
            return result
        except Exception as e:
            self._release_models_after_failure()
            raise map_engine_error(e, context=f"Standard Search [{query.prompt}]")

    def search_verified(self, query: SearchQuery) -> SearchResult:
        self._enforce_online_state()
        try:
            search_id = uuid.uuid4()
            result = self._live_verified_pipeline.execute_verified_search(search_id, query)
            self._resolve_physical_paths(result)
            return result
        except Exception as e:
            self._release_models_after_failure()
            raise map_engine_error(e, context=f"Verified Search [{query.prompt}]")

    def chat(self, query: SearchQuery) -> QAResponse:
        self._enforce_online_state()
        try:
            search_id = uuid.uuid4()
            result = self._live_qa_pipeline.execute_chat(search_id, query)
            self._resolve_citation_paths(result)
            return result
        except Exception as e:
            self._release_models_after_failure()
            raise map_engine_error(e, context=f"Chat QA Generation [{query.prompt}]")

    def _release_models_after_failure(self) -> None:
        """A failed live call must not leave a model resident for the next request to wait on."""
        try:
            self._async_service.release_all()
        except Exception as e:
            logger.error(f"Failed to release models after live inference failure: {e}")

    def _enforce_online_state(self) -> None:
        with self._state_lock:
            if self._state == EngineState.OFFLINE:
                raise EngineOfflineError("Engine is offline.")
            if self._state == EngineState.INGESTING:
                raise EngineBusyError("Hardware is locked for batch ingestion. Inference paused.")

    def _resolve_physical_paths(self, search_result: SearchResult) -> None:
        """Hydrates chunks with physical file paths before sending to the UI."""
        for chunk in search_result.results:
            try:
                if chunk.extracted_file_id:
                    path = self._storage_api.get_file_path(chunk.extracted_file_id)
                    object.__setattr__(chunk, 'physical_path', str(path))
                else:
                    source = self._storage_api.get_source(chunk.source_id)
                    if source and source.file:
                        object.__setattr__(chunk, 'physical_path', str(source.file.file_path))
            except Exception as e:
                logger.warning(f"Failed to resolve physical path for chunk {chunk.chunk_id}: {e}")

    def _resolve_citation_paths(self, response: QAResponse) -> None:
        """Hydrates citation evidence with physical file paths before sending to the UI."""
        for citation in response.citations:
            try:
                physical_path = getattr(citation, 'physical_path', None)
                source_id_raw = getattr(citation, 'source_id', None)

                if not physical_path and source_id_raw:
                    # Convert to strict UUID to satisfy type checker
                    valid_source_id = source_id_raw if isinstance(source_id_raw, uuid.UUID) else uuid.UUID(
                        str(source_id_raw))
                    source = self._storage_api.get_source(valid_source_id)

                    if source and source.file:
                        object.__setattr__(citation, 'physical_path', str(source.file.file_path))
            except Exception as e:
                chunk_id = getattr(citation, 'chunk_id', 'Unknown')
                logger.warning(f"Failed to resolve physical path for citation {chunk_id}: {e}")

    # ==========================================
    # UTILITIES & CLI
    # ==========================================

    @staticmethod
    def _aggregate_metrics(
            pipeline_results: Dict[PipelinePhase, PhaseBatchResult],
            error_dict: Dict[str, str],
            metrics_dict: Dict[str, int]
    ) -> None:
        if not pipeline_results: return

        if PipelinePhase.SCAN in pipeline_results:
            metrics_dict["scanned"] += pipeline_results[PipelinePhase.SCAN].successful_count
        if PipelinePhase.RAG_INDEXING in pipeline_results:
            rag_res = pipeline_results[PipelinePhase.RAG_INDEXING]
            metrics_dict["success"] += rag_res.successful_count
            metrics_dict["failed"] += rag_res.failed_count
            for item in rag_res.items:
                if getattr(item, "success", False):
                    metrics_dict["vectors"] += getattr(item, "dense_vectors_saved", 0)
        if PipelinePhase.CLIP_EXTRACTION in pipeline_results:
            clip_res = pipeline_results[PipelinePhase.CLIP_EXTRACTION]
            for item in clip_res.items:
                if getattr(item, "success", False):
                    metrics_dict["clips"] += getattr(item, "extracted_clips_count", 0)

        for phase, result in pipeline_results.items():
            if result.status in [PhaseStatus.FAILED, PhaseStatus.PARTIAL_SUCCESS]:
                for idx, err_str in enumerate(result.errors):
                    error_dict[f"{phase.value}_error_{idx}"] = err_str

    @staticmethod
    def _build_report(target_path: str, metrics: dict, errors: dict, duration: float) -> IngestionReport:
        return IngestionReport(
            target_path=target_path, total_files_scanned=metrics["scanned"],
            successful_ingestions=metrics["success"], failed_ingestions=metrics["failed"],
            errors=errors, total_duration_seconds=round(duration, 2),
            vectors_created=metrics["vectors"], clips_extracted=metrics["clips"]
        )

    def interactive_cli(self) -> None:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=======================================================")
            print("          O N Y X   E N T E R P R I S E   A I          ")
            print("=======================================================\n")
            print("1. File Ingestion [Video Pipeline]")
            print("2. File Ingestion [Audio Pipeline]")
            print("3. Fast Retrieval [Standard RAG]")
            print("4. Context-Aware Retrieval [Verified SLM Scrubber]")
            print("5. Chat with Data [RAG -> LLM Synthesis]")
            print("6. Shutdown Engine")

            choice = input("\nChoice (1-6): ").strip()

            try:
                if choice == "1":
                    path = input("Enter target directory path for Video Ingestion: ").strip()
                    report = self.ingest_videos(path)
                    print(
                        f"\n[Complete] Success: {report.successful_ingestions} | Failed: {report.failed_ingestions} | Vectors: {report.vectors_created}")
                    if report.errors: print(f"[Errors]: {report.errors}")
                    input("\nPress Enter...")
                elif choice == "2":
                    path = input("Enter target directory path for Audio Ingestion: ").strip()
                    report = self.ingest_audios(path)
                    print(
                        f"\n[Complete] Success: {report.successful_ingestions} | Failed: {report.failed_ingestions} | Vectors: {report.vectors_created}")
                    if report.errors: print(f"[Errors]: {report.errors}")
                    input("\nPress Enter...")
                elif choice == "3":
                    prompt = input("Enter your search query: ").strip()
                    results = self.search_standard(SearchQuery(prompt=prompt))
                    for idx, chunk in enumerate(results.results, 1):
                        print(
                            f"\n--- Result {idx} ---\nPath: {getattr(chunk, 'physical_path', 'Unknown')}\nContent: {chunk.text_payload}")
                    input("\nPress Enter...")
                elif choice == "4":
                    prompt = input("Enter your high-precision search query: ").strip()
                    results = self.search_verified(SearchQuery(prompt=prompt))
                    for idx, chunk in enumerate(results.results, 1):
                        print(
                            f"\n--- Verified Result {idx} ---\nPath: {getattr(chunk, 'physical_path', 'Unknown')}\nContent: {chunk.text_payload}")
                    input("\nPress Enter...")
                elif choice == "5":
                    prompt = input("Enter your question: ").strip()
                    response = self.chat(SearchQuery(prompt=prompt))
                    print(f"\n--- Onyx AI ---\n{response.generated_answer}")
                    print(f"\n[Citations Used: {len(response.citations)}]")
                    for idx, citation in enumerate(response.citations, 1):
                        print(f"  {idx}. {getattr(citation, 'physical_path', 'Unknown')} (Doc ID: {citation.doc_id})")
                    input("\nPress Enter...")
                elif choice == "6":
                    self.stop()
                    sys.exit(0)
            except Exception as e:
                print(f"\n[ERROR]: {str(e)}")
                input("\nPress Enter...")