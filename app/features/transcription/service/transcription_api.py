# filename: transcription_api_v1.py
# path: app/features/transcription/service/transcription_api.py
import uuid

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
    """
    The central brain of the Transcription Feature.
    Orchestrates the retrieval of the VAD map, VRAM hardware management,
    and the execution of the CTranslate2 model.
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

    def process_audio(self, source_id: uuid.UUID, audio_path: str, release_after: bool = True) -> Transcript:
        """
        Takes a raw audio file and its corresponding VAD map to generate a persisted Transcript.
        Guarantees VRAM safety and idempotency.

        Args:
            source_id: The UUID of the original media source.
            audio_path: Physical OS path to the extracted audio file.
            release_after: If True, explicitly unloads the model from VRAM immediately after processing.
        """
        # 1. Idempotency Check: Return early if already transcribed
        existing_transcript = self._repo.get_by_source_id(source_id)
        if existing_transcript:
            return existing_transcript

        # 2. Fetch the prerequisite VAD Map
        vad_map = self._vad_repo.get_map_by_source_id(source_id)
        if not vad_map:
            raise TranscriptionAudioError(
                f"Cannot transcribe Source {source_id}: VAD Map not found. "
                "The VAD phase must successfully complete before transcription."
            )

        # 3. Define the VRAM requirements for the Orchestrator
        manifest = ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path=settings.WHISPER_MODEL_PATH,
            required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
        )

        loaded_model = None
        try:
            # 4. Safely request the GPU resource
            loaded_model = self._orchestrator.acquire_model(manifest, self._adapter)

            # 5. Execute the transcription adapter logic
            transcript = self._adapter.transcribe(
                model=loaded_model,
                audio_path=audio_path,
                vad_map=vad_map,
                source_id=source_id
            )

            # 6. Save the resulting JSONB document to the database
            saved_transcript = self._repo.save(transcript)

            return saved_transcript

        except Exception as e:
            raise map_transcription_error(e) from e

        finally:
            # 7. Respect the pipeline's lifecycle rules
            if loaded_model and release_after:
                self._orchestrator.release_model(loaded_model, self._adapter, flush=True)