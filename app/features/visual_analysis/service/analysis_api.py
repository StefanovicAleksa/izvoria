# Canvas: visual_analysis_api_v3
# Path: app/features/visual_analysis/service/analysis_api.py

import logging
import uuid
from typing import List, Set

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType

from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames
from app.features.storage.service.storage_api import StorageAPI

from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis, FrameAnalysis
from app.features.visual_analysis.domain.interfaces import IVisualAnalysisRepository, IVisualLanguageModelAdapter
from app.features.visual_analysis.utils.error_mapper import map_visual_analysis_error
from app.features.visual_analysis.utils.grounding_check import (
    drop_items_already_reported,
    drop_marks_copied_from_other_frames,
    ground_observation
)
from app.features.visual_analysis.utils.observation_renderer import render_observation

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

        # Strict Domain Manifest Isolation for Vision Analysis (Phase 12)
        self._manifest = ModelManifest(
            model_type=ModelType.VISION,
            model_path=settings.VISION_MODEL_PATH,
            required_vram_mb=settings.VISION_VRAM_REQUIREMENT_MB
        )

    @property
    def manifest(self) -> ModelManifest:
        """Exposes the specific VLM manifest to the orchestrator for background preloading."""
        return self._manifest

    @property
    def adapter(self) -> IVisualLanguageModelAdapter:
        """Exposes the internal adapter to the orchestrator for standardized loading/unloading."""
        return self._vlm_adapter

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
        multiple chunks without reloading the massive 7B+ weights every time.
        """
        try:
            # 1. Idempotency Check: Don't re-run expensive VLM inference if we survived a crash
            existing_record = self._repo.get_by_chunk_id(chunk.chunk_id)
            if existing_record:
                logger.info(f"Visual analysis already exists for chunk {chunk.chunk_id}. Skipping.")
                return existing_record

            # 2. Edge Case: The visual cue detector flagged it, but no frames survived extraction
            if not extracted_frames.frames:
                return self._save_empty_result(source_id, chunk)

            frame_analyses: List[FrameAnalysis] = []
            reported_items: Set[str] = set()
            frames = sorted(extracted_frames.frames, key=lambda f: f.timestamp_sec)
            frames_per_call = max(1, settings.VISION_MAX_IMAGES_PER_CALL)

            # 3. Read the frames chronologically, several per multimodal call: the shared prompt
            # (system rules + chunk text) is encoded once per group instead of once per frame.
            # Only frames with visible evidence (legible text or drawn marks) get a description.
            for group_start in range(0, len(frames), frames_per_call):
                group = frames[group_start:group_start + frames_per_call]
                image_paths = [self._storage.get_file_path(frame.file_id) for frame in group]

                logger.info(f"  -> VLM analyzing {len(group)} frame(s) for Chunk {chunk.chunk_id}...")

                observations = self._vlm_adapter.analyze_frames(
                    model=vlm_model,
                    image_paths=image_paths,
                    original_text=chunk.original_text,
                    context_string=chunk.context_string
                )

                grounded_group = drop_marks_copied_from_other_frames(
                    [ground_observation(observation, chunk.original_text) for observation in observations]
                )

                for frame, grounded in zip(group, grounded_group):
                    if grounded is not None:
                        grounded = drop_items_already_reported(grounded, reported_items)
                    if grounded is None:
                        logger.info(f"Frame {frame.frame_id} shows no legible text or marks; no description stored.")
                        continue

                    frame_analyses.append(
                        FrameAnalysis(
                            frame_id=frame.frame_id,
                            description=render_observation(grounded),
                            timestamp_sec=frame.timestamp_sec
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
