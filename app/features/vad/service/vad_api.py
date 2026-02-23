# File: app/features/vad/service/vad_api.py
import uuid

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.vad.domain.entities import VadMap
from app.features.vad.domain.interfaces import IVadRepository, IVadModelAdapter
from app.features.vad.utils.error_mapper import map_vad_error


class VadAPI:
    """
    The central brain of the VAD Feature.
    Orchestrates the database, the strict VRAM hardware manager, and the AI model.
    """

    def __init__(
            self,
            repository: IVadRepository,
            orchestrator: ModelManagerOrchestrator,
            adapter: IVadModelAdapter
    ):
        self._repo = repository
        self._orchestrator = orchestrator
        self._adapter = adapter

    def process_audio(self, source_id: uuid.UUID, audio_path: str) -> VadMap:
        """
        Takes a raw audio file and returns a persisted virtual map of human speech.
        Guarantees VRAM safety and idempotency.
        """
        existing_map = self._repo.get_map_by_source_id(source_id)
        if existing_map:
            return existing_map

        # Use the dynamic settings for the manifest
        manifest = ModelManifest(
            model_type=ModelType.VAD,
            model_path=settings.VAD_MODEL_PATH,
            required_vram_mb=settings.VAD_VRAM_REQUIREMENT_MB
        )

        loaded_model = None
        try:
            loaded_model = self._orchestrator.acquire_model(manifest, self._adapter)

            vad_map = self._adapter.generate_vad_map(
                model=loaded_model,
                audio_path=audio_path,
                source_id=source_id
            )

            saved_map = self._repo.save_map(vad_map)

            return saved_map

        except Exception as e:
            raise map_vad_error(e) from e

        finally:
            if loaded_model:
                self._orchestrator.release_model(loaded_model, self._adapter, flush=True)