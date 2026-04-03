# Canvas: detection_api_v5
# Path: app/features/visual_cue_detection/service/detection_api.py

import logging
from typing import Optional

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.visual_cue_detection.domain.interfaces import IVisualCueDetectorAdapter
from app.features.contextual_retrieval.domain.interfaces import IContextualRepository
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.visual_cue_detection.utils.error_mapper import map_detection_error

logger = logging.getLogger(__name__)


class VisualCueDetectionAPI:
    def __init__(
            self,
            orchestrator: ModelManagerOrchestrator,
            adapter: IVisualCueDetectorAdapter,
            repository: IContextualRepository
    ):
        self._orchestrator = orchestrator
        self._adapter = adapter
        self._repository = repository

        # Strict Domain Manifest Isolation
        self._manifest = ModelManifest(
            model_type=ModelType.VISUAL_CUE_DETECTION,
            model_path=settings.VISUAL_CUE_DETECTION_MODEL_PATH,
            required_vram_mb=settings.VISUAL_CUE_DETECTION_VRAM_REQUIREMENT_MB
        )

    @property
    def adapter(self) -> IVisualCueDetectorAdapter:
        return self._adapter

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    def process_transcript(
            self,
            transcript: ContextualizedTranscript,
            model: Optional[LoadedModel] = None,
            release_after: bool = True
    ) -> ContextualizedTranscript:
        loaded_model = model
        manage_lifecycle = loaded_model is None

        try:
            if manage_lifecycle:
                loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter)

            # =====================================================================
            # THE FIX: Execute the visual cue chunk evaluation via the AI adapter
            # without silently swallowing errors. If a chunk fails, the job fails.
            # =====================================================================
            updated_chunks = []
            for chunk in transcript.chunks:
                try:
                    updated_chunk = self._adapter.detect_cues(loaded_model, chunk)
                    updated_chunks.append(updated_chunk)
                except Exception as e:
                    # Bubble up the exact error so the JobManager marks it as FAILED
                    raise map_detection_error(e, context=f"Detecting cues for chunk {chunk.chunk_id}") from e

            # Assign updated chunks back to the transcript
            transcript.chunks = updated_chunks

            # Persist the evaluated flags to the database
            self._repository.update(transcript)

            return transcript

        except Exception as e:
            raise map_detection_error(e,
                                      context=f"VisualCueDetectionAPI.process_transcript for Source {transcript.source_id}") from e

        finally:
            # Respect the batch lifecycle lock requested by the Phase Orchestrator
            if manage_lifecycle and loaded_model and release_after:
                self._orchestrator.release_model(loaded_model, self._adapter, flush=True)