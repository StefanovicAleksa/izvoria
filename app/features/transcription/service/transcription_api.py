# Canvas: transcription_api_v6
# Path: app/features/transcription/service/transcription_api.py

import uuid
from typing import Optional

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.transcription.domain.interfaces import ITranscriptionAdapter, ITranscriptRepository
from app.features.vad.domain.interfaces import IVadRepository
from app.features.transcription.domain.entities import Transcript
from app.features.transcription.utils.error_mapper import map_transcription_error


class TranscriptionAPI:
    """
    Orchestrates the conversion of audio files into raw text transcripts
    using Voice Activity Detection maps to bypass silent sections.
    """

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

    def process_audio(
            self,
            source_id: uuid.UUID,
            audio_path: str,
            language: Optional[str] = None,
            hotwords: Optional[str] = None,
            initial_prompt: Optional[str] = None, # FIX: Added to signature
            release_after: bool = False
    ) -> Transcript:

        # 1. Idempotency Check
        existing_transcript = self._repo.get_by_source_id(source_id)
        if existing_transcript:
            return existing_transcript

        # 2. Fetch VAD Map
        vad_map = self._vad_repo.get_map_by_source_id(source_id)
        if not vad_map or not vad_map.segments:
            return self._repo.save(Transcript(source_id=source_id, segments=[]))

        # 3. Determine Language
        target_language = language or settings.APPLIANCE_LANGUAGE

        # 4. Load Model
        manifest = ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path=settings.WHISPER_MODEL_PATH,
            required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
        )
        loaded_model = self._orchestrator.acquire_model(manifest, self._adapter)

        try:
            # 5. Transcribe with Domain Hints and Language Constraints
            segments = self._adapter.transcribe(
                model=loaded_model,
                audio_path=audio_path,
                vad_map=vad_map,
                language=target_language,
                hotwords=hotwords,
                initial_prompt=initial_prompt # FIX: Passed down to FasterWhisperAdapter
            )

            transcript = Transcript(source_id=source_id, segments=segments)
            return self._repo.save(transcript)

        except Exception as e:
            raise map_transcription_error(e) from e
        finally:
            if release_after:
                self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

    def get_transcript(self, source_id: uuid.UUID) -> Optional[Transcript]:
        return self._repo.get_by_source_id(source_id)

    def update_transcript(self, transcript: Transcript) -> Transcript:
        return self._repo.update(transcript)