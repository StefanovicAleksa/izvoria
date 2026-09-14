# Canvas: diarization_api_v4
# Path: app/features/diarization/service/diarization_api.py

import uuid

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.storage.service.storage_api import StorageAPI
from app.features.vad.domain.interfaces import IVadRepository
from app.features.transcription.service.transcription_api import TranscriptionAPI

from app.features.diarization.domain.entities import DiarizationMap
from app.features.diarization.domain.interfaces import IDiarizationRepository, IDiarizationModelAdapter
from app.features.diarization.utils.aligner import align_transcript_with_diarization
from app.features.diarization.utils.error_mapper import map_diarization_error
from app.features.diarization.exceptions.model import DiarizationManifestError


class DiarizationAPI:
    def __init__(
            self,
            repository: IDiarizationRepository,
            vad_repository: IVadRepository,
            transcription_api: TranscriptionAPI,
            storage_api: StorageAPI,
            orchestrator: ModelManagerOrchestrator,
            adapter: IDiarizationModelAdapter
    ):
        self._repo = repository
        self._vad_repo = vad_repository
        self._transcription_api = transcription_api
        self._storage = storage_api
        self._orchestrator = orchestrator
        self._adapter = adapter

    def process_audio(self, source_id: uuid.UUID, audio_path: str, release_after: bool = True) -> DiarizationMap:
        # 1. Idempotency Check (the saved map is the completion marker, see step 8)
        existing_map = self._repo.get_map_by_source_id(source_id)
        if existing_map:
            return existing_map

        # 2. Fetch Prerequisites
        vad_map = self._vad_repo.get_map_by_source_id(source_id)
        if not vad_map:
            raise DiarizationManifestError(f"Cannot diarize Source {source_id}: VAD Map not found.")

        transcript = self._transcription_api.get_transcript(source_id)
        if not transcript:
            raise DiarizationManifestError(f"Cannot diarize Source {source_id}: Whisper Transcript not found.")

        # 3. Define VRAM Requirements (Strict Config Inheritance)
        manifest = ModelManifest(
            model_type=ModelType.DIARIZATION,
            model_path=settings.DIARIZATION_MODEL_PATH,
            required_vram_mb=settings.DIARIZATION_VRAM_REQUIREMENT_MB
        )

        loaded_model = None
        try:
            # 4. Safely request the GPU resource
            loaded_model = self._orchestrator.acquire_model(manifest, self._adapter)

            # 5. Execute NeMo Diarization
            diarization_map = self._adapter.generate_diarization_map(
                model=loaded_model,
                audio_path=audio_path,
                vad_map=vad_map,
                source_id=source_id
            )

            # 6. Execute the Aligner Math
            aligned_transcript = align_transcript_with_diarization(transcript, diarization_map)

            # 7. Update the Transcription Record
            self._transcription_api.update_transcript(aligned_transcript)

            # 8. Save the Diarization Map LAST. It doubles as the idempotency marker, so saving it
            # before the transcript update meant a crash in between skipped alignment forever.
            return self._repo.save_map(diarization_map)

        except Exception as e:
            raise map_diarization_error(e) from e

        finally:
            if loaded_model and release_after:
                self._orchestrator.release_model(loaded_model, self._adapter, flush=True)
