# Canvas: onyx_engine_v4
# Path: app/engine/service/onyx_engine.py

import logging
import threading
import time
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
from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult

from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline
from app.orchestration.service.pipelines.batch_audio_pipeline import BatchAudioPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult

logger = logging.getLogger(__name__)


class OnyxEngine(IOnyxEngine):
    def __init__(
            self,
            async_model_service: AsyncModelService,
            media_pipeline: BatchMediaPipeline,
            audio_pipeline: BatchAudioPipeline,
            retrieval_api: RetrievalAPI,
            storage_api: StorageAPI
    ):
        self._async_service = async_model_service
        self._media_pipeline = media_pipeline
        self._audio_pipeline = audio_pipeline
        self._retrieval_api = retrieval_api
        self._storage_api = storage_api

        self._state = EngineState.OFFLINE
        self._state_lock = threading.RLock()

    def start(self) -> None:
        with self._state_lock:
            if self._state != EngineState.OFFLINE:
                return

            self._state = EngineState.BOOTING
            logger.info("Booting Onyx Engine...")

            try:
                # Ensure DB Tables exist cleanly by getting the bind from a transient session
                with db_manager.session() as session:
                    engine = session.get_bind()
                    Base.metadata.create_all(bind=engine)

                self._state = EngineState.ONLINE
                logger.info("Onyx Engine is ONLINE and ready.")

            except Exception as e:
                self._state = EngineState.ERROR
                raise map_engine_error(e, context="Starting Engine")

    def stop(self) -> None:
        with self._state_lock:
            if self._state == EngineState.OFFLINE:
                return

            logger.info("Shutting down Onyx Engine...")
            try:
                # Safely kill the background model loader thread
                self._async_service.shutdown()

                # Close the database connection pool
                db_manager.dispose()

                self._state = EngineState.OFFLINE
                logger.info("Onyx Engine is OFFLINE.")
            except Exception as e:
                self._state = EngineState.ERROR
                raise map_engine_error(e, context="Stopping Engine")

    def get_state(self) -> EngineState:
        with self._state_lock:
            return self._state

    def ingest(self, target_path: str) -> IngestionReport:
        with self._state_lock:
            if self._state == EngineState.OFFLINE:
                raise EngineOfflineError("Cannot ingest media: Engine is offline.")
            if self._state == EngineState.INGESTING:
                raise EngineBusyError("Cannot ingest media: Engine is already processing a batch.")

            self._state = EngineState.INGESTING

        logger.info(f"Starting Ingestion for path: {target_path}")
        start_time = time.time()

        aggregated_errors: Dict[str, str] = {}
        metrics: Dict[str, int] = {
            "scanned": 0,
            "success": 0,
            "failed": 0,
            "vectors": 0,
            "clips": 0
        }

        try:
            # 1. Run Audio Pipeline
            audio_results = self._audio_pipeline.process_directory(target_path)
            self._aggregate_metrics(audio_results, aggregated_errors, metrics)

            # 2. Run Video Pipeline
            video_results = self._media_pipeline.process_directory(target_path, SourceType.VIDEO)
            self._aggregate_metrics(video_results, aggregated_errors, metrics)

        except Exception as e:
            raise map_engine_error(e, context=f"Ingesting {target_path}")
        finally:
            with self._state_lock:
                self._state = EngineState.ONLINE

        duration = time.time() - start_time

        return IngestionReport(
            target_path=target_path,
            total_files_scanned=metrics["scanned"],
            successful_ingestions=metrics["success"],
            failed_ingestions=metrics["failed"],
            errors=aggregated_errors,
            total_duration_seconds=round(duration, 2),
            vectors_created=metrics["vectors"],
            clips_extracted=metrics["clips"]
        )

    def search(self, query: SearchQuery) -> SearchResult:
        with self._state_lock:
            if self._state == EngineState.OFFLINE:
                raise EngineOfflineError("Cannot execute search: Engine is offline.")
            if self._state == EngineState.INGESTING:
                raise EngineBusyError("Cannot execute search: VRAM is currently locked for media ingestion.")

        try:
            raw_result = self._retrieval_api.search(query)

            for chunk in raw_result.results:
                if chunk.extracted_file_id:
                    try:
                        physical_path = self._storage_api.get_file_path(chunk.extracted_file_id)
                        object.__setattr__(chunk, 'physical_path', str(physical_path))
                    except Exception as e:
                        logger.warning(f"Could not resolve physical path for clip {chunk.extracted_file_id}: {e}")
                else:
                    source = self._storage_api.get_source(chunk.source_id)
                    if source and source.file:
                        object.__setattr__(chunk, 'physical_path', str(source.file.file_path))

            return raw_result

        except Exception as e:
            raise map_engine_error(e, context=f"Searching for '{query.prompt}'")

    @staticmethod
    def _aggregate_metrics(
            pipeline_results: Dict[PipelinePhase, PhaseBatchResult],
            error_dict: Dict[str, str],
            metrics_dict: Dict[str, int]
    ) -> None:
        """Helper to extract vital stats from the massive pipeline payload."""
        if not pipeline_results:
            return

        # Scan Phase dictates total files
        if PipelinePhase.SCAN in pipeline_results:
            metrics_dict["scanned"] += pipeline_results[PipelinePhase.SCAN].successful_count

        # RAG Indexing dictates total success
        if PipelinePhase.RAG_INDEXING in pipeline_results:
            rag_res = pipeline_results[PipelinePhase.RAG_INDEXING]
            metrics_dict["success"] += rag_res.successful_count
            metrics_dict["failed"] += rag_res.failed_count

            for item in rag_res.items:
                if getattr(item, "success", False):
                    metrics_dict["vectors"] += getattr(item, "dense_vectors_saved", 0)

        # Clip extraction metrics
        if PipelinePhase.CLIP_EXTRACTION in pipeline_results:
            clip_res = pipeline_results[PipelinePhase.CLIP_EXTRACTION]
            for item in clip_res.items:
                if getattr(item, "success", False):
                    metrics_dict["clips"] += getattr(item, "extracted_clips_count", 0)

        # Extract generic errors across all phases
        for phase, result in pipeline_results.items():
            if result.status in [PhaseStatus.FAILED, PhaseStatus.PARTIAL_SUCCESS]:
                for idx, err_str in enumerate(result.errors):
                    error_dict[f"{phase.value}_error_{idx}"] = err_str