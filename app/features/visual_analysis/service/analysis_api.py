# Canvas: visual_analysis_api_v1
# Path: app/features/visual_analysis/service/analysis_api.py

import logging
import uuid
from typing import List

from app.core.model_manager.domain.entities import LoadedModel
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames
from app.features.storage.service.storage_api import StorageAPI

from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis, FrameAnalysis
from app.features.visual_analysis.domain.interfaces import IVisualAnalysisRepository, IVisualLanguageModelAdapter
from app.features.visual_analysis.utils.error_mapper import map_visual_analysis_error

logger = logging.getLogger(__name__)


class VisualAnalysisAPI:
    """
    Orchestrates the strictly VRAM-bound analysis of physical video frames.
    Pulls images from the Storage Vault, applies grounded text context,
    and persists the semantic descriptions for RAG ingestion.
    """

    def __init__(
        self,
        repository: IVisualAnalysisRepository,
        vlm_adapter: IVisualLanguageModelAdapter,
        storage_api: StorageAPI
    ):
        self._repo = repository
        self._vlm_adapter = vlm_adapter
        self._storage = storage_api

    def process_chunk(
        self,
        source_id: uuid.UUID,
        chunk: ContextualizedChunk,
        extracted_frames: ExtractedChunkFrames,
        vlm_model: LoadedModel
    ) -> ChunkVisualAnalysis:
        """
        Executes the Vision-Language Model on all vaulted frames for a given chunk.
        The `vlm_model` is passed in to allow the parent pipeline to batch-process
        multiple chunks without reloading the 4.5GB weights every time.
        """
        try:
            # 1. Idempotency Check: Don't re-run expensive VLM inference if we survived a crash
            existing_record = self._repo.get_by_chunk_id(chunk.chunk_id)
            if existing_record:
                logger.info(f"Visual analysis already exists for chunk {chunk.chunk_id}. Skipping.")
                return existing_record

            # 2. Edge Case: The visual cue detector flagged it, but all frames were too blurry
            if not extracted_frames.frames:
                return self._save_empty_result(source_id, chunk)

            frame_analyses: List[FrameAnalysis] = []

            # 3. Process each vaulted frame chronologically
            for frame in extracted_frames.frames:
                # Retrieve the physical file path from the secure Onyx Storage Vault
                physical_path = self._storage.get_file_path(frame.file_id)

                logger.info(f"  -> VLM Analyzing Frame {frame.frame_id} for Chunk {chunk.chunk_id}...")

                # Execute the multimodal inference, strictly bounded by the chunk's text
                description = self._vlm_adapter.analyze_frame(
                    model=vlm_model,
                    image_path=physical_path,
                    original_text=chunk.original_text,
                    context_string=chunk.context_string
                )

                frame_analyses.append(
                    FrameAnalysis(
                        frame_id=frame.frame_id,
                        description=description
                    )
                )

            # 4. Aggregate and Persist Mapping
            visual_analysis = ChunkVisualAnalysis(
                source_id=source_id,
                chunk_id=chunk.chunk_id,
                analyses=frame_analyses
            )

            return self._repo.save(visual_analysis)

        except Exception as e:
            raise map_visual_analysis_error(e, context=f"VisualAnalysisAPI.process_chunk for {chunk.chunk_id}") from e

    def _save_empty_result(self, source_id: uuid.UUID, chunk: ContextualizedChunk) -> ChunkVisualAnalysis:
        """Saves an empty ledger record if there were no valid frames to analyze."""
        empty_analysis = ChunkVisualAnalysis(
            source_id=source_id,
            chunk_id=chunk.chunk_id,
            analyses=[]
        )
        return self._repo.save(empty_analysis)