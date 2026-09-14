# Canvas: visual_analysis_interfaces_v2
# Path: app/features/visual_analysis/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis


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
    ) -> List[str]:
        """
        Describes a chunk's frames (in chronological order) in one multimodal call, using the surrounding
        text only to decide what to look at.
        Returns exactly one description per image, in order. An entry is empty when the model produced
        nothing usable for that frame.
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
