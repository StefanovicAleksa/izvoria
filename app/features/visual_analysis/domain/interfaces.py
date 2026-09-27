# Canvas: visual_analysis_interfaces_v2
# Path: app/features/visual_analysis/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis, FrameObservation


class IVisualLanguageModelAdapter(IModelLoader, ABC):
    """
    Contract for the multi-modal Vision-Language Model (e.g., Qwen-VL, LLaVA).
    Inherits from IModelLoader to support strict VRAM management via the Orchestrator.
    """

    @abstractmethod
    def analyze_frames(
        self,
        model: LoadedModel,
        image_paths: List[Path],
        original_text: str,
        context_string: str
    ) -> List[Optional[FrameObservation]]:
        """
        Reads a chunk's frames (in chronological order) in one multimodal call, using the surrounding text only to
        decide what to look at.
        Returns exactly one observation per image, in order. An entry is None when the model produced nothing
        structured for that frame.
        """
        pass


    @abstractmethod
    def describe_image(self, model: LoadedModel, image_path: Path) -> Optional[FrameObservation]:
        """
        Reads one standalone image with no surrounding text: what it shows, the text legible in it, its marks.
        None when the model produced nothing structured.
        """
        pass


class IVisualAnalysisRepository(ABC):
    """
    Contract for persisting the ChunkVisualAnalysis ledger to PostgreSQL.
    Critical for crash-recovery and idempotency.
    """

    @abstractmethod
    def save(self, visual_analysis: ChunkVisualAnalysis) -> ChunkVisualAnalysis:
        """Saves the completed analysis aggregate to the database."""
        pass

    @abstractmethod
    def get_by_chunk_id(self, chunk_id: uuid.UUID) -> Optional[ChunkVisualAnalysis]:
        """Checks if a chunk has already been analyzed to prevent duplicate VLM inference."""
        pass
