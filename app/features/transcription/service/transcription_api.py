# Canvas: transcription_api_v3
# Path: app/features/transcription/service/transcription_api.py

import uuid
from typing import Optional

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.vad.domain.interfaces import IVadRepository
from app.features.transcription.domain.entities import Transcript
from app.features.transcription.domain.interfaces import ITranscriptRepository, ITranscriptionAdapter
from app.features.transcription.utils.error_mapper import map_transcription_error
from app.features.transcription.exceptions.model import TranscriptionAudioError


class TranscriptionAPI:
    def __init__(
            self,
            repository: ITranscriptRepository,
            vad_repository: IVadRepository,
            orchestrator: ModelManagerOrchestrator,
            adapter: ITranscriptionAdapter
    ):
        self._repo = repository
        self._vad_repo = vad_repository
        self._orchestrator = orchestrator
        self._adapter = adapter

    def get_transcript(self, source_id: uuid.UUID) -> Optional[Transcript]:
        """Public API method for other features to retrieve an existing Transcript."""
        try:
            return self._repo.get_by_source_id(source_id)
        except Exception as e:
            raise map_transcription_error(e) from e

    def update_transcript(self, transcript: Transcript) -> Transcript:
        """
        Public API method for other features (like Diarization) to update
        an existing transcript with new enriched data (e.g., Speaker IDs).
        """
        try:
            return self._repo.update(transcript)
        except Exception as e:
            raise map_transcription_error(e) from e

    def process_audio(self, source_id: uuid.UUID, audio_path: str, release_after: bool = True) -> Transcript:
        existing_transcript = self._repo.get_by_source_id(source_id)
        if existing_transcript:
            return existing_transcript

        vad_map = self._vad_repo.get_map_by_source_id(source_id)
        if not vad_map:
            raise TranscriptionAudioError(
                f"Cannot transcribe Source {source_id}: VAD Map not found. "
                "The VAD phase must successfully complete before transcription."
            )

        manifest = ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path=settings.WHISPER_MODEL_PATH,
            required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
        )

        loaded_model = None
        try:
            loaded_model = self._orchestrator.acquire_model(manifest, self._adapter)

            transcript = self._adapter.transcribe(
                model=loaded_model,
                audio_path=audio_path,
                vad_map=vad_map,
                source_id=source_id
            )

            saved_transcript = self._repo.save(transcript)
            return saved_transcript

        except Exception as e:
            raise map_transcription_error(e) from e

        finally:
            if loaded_model and release_after:
                self._orchestrator.release_model(loaded_model, self._adapter, flush=True)