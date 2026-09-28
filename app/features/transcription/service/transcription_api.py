# Canvas: transcription_api_v7
# Path: app/features/transcription/service/transcription_api.py

import logging
import uuid
from typing import Optional

from app.core.config import settings
from app.core.language.domain.entities import LanguagePack
from app.core.language.service.language_api import get_language_pack
from app.core.language.utils.script import normalize_script
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.transcription.domain.interfaces import ITranscriptionAdapter, ITranscriptRepository
from app.features.vad.domain.interfaces import IVadRepository
from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.features.transcription.exceptions import LanguageMismatchError
from app.features.transcription.utils.error_mapper import map_transcription_error
from app.features.transcription.utils.prompt_echo_filter import remove_prompt_echoes

logger = logging.getLogger(__name__)


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
        pack = get_language_pack()
        target_language = language or pack.transcription.whisper_language

        # 4. Load Model
        manifest = ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path=settings.WHISPER_MODEL_PATH,
            required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
        )
        loaded_model = self._orchestrator.acquire_model(manifest, self._adapter)

        try:
            # 5. Refuse a file in another language before forcing Whisper's language on it
            self._check_language(loaded_model, source_id, audio_path, vad_map, pack)

            # 6. Transcribe with Domain Hints and Language Constraints
            segments = self._adapter.transcribe(
                model=loaded_model,
                audio_path=audio_path,
                vad_map=vad_map,
                language=target_language,
                hotwords=hotwords,
                initial_prompt=initial_prompt # FIX: Passed down to FasterWhisperAdapter
            )

            # 7. Whisper sometimes repeats the text it was primed with (hotwords/prompt) as if it had been said;
            # that text must never reach the index.
            segments, echoed_words = remove_prompt_echoes(
                segments, [hotwords, initial_prompt], settings.WHISPER_PROMPT_ECHO_MIN_TOKENS
            )
            if echoed_words:
                logger.warning(
                    f"Removed {echoed_words} word(s) Whisper copied from its hotwords/prompt for source {source_id}."
                )

            # 8. One script for the whole index (Serbian: Cyrillic -> Latin), word by word so timestamps still line up
            if pack.transcription.script:
                segments = [_normalize_segment_script(segment, pack.transcription.script) for segment in segments]

            transcript = Transcript(source_id=source_id, segments=segments)
            return self._repo.save(transcript)

        except Exception as e:
            raise map_transcription_error(e) from e
        finally:
            if release_after:
                self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

    def _check_language(self, loaded_model, source_id: uuid.UUID, audio_path: str, vad_map, pack: LanguagePack) -> None:
        """
        Raises LanguageMismatchError when Whisper is confident the speech is in a language this appliance doesn't
        serve. An unsure detection passes: short or noisy speech is misidentified often, and rejecting a correct
        file loses real evidence, while the check exists only to stop confidently foreign files.
        """
        detection = self._adapter.detect_language(loaded_model, audio_path, vad_map)
        if detection is None:
            return

        detected, probability = detection
        rules = pack.transcription
        if detected in rules.accepted_detections:
            return
        if probability < rules.min_detection_probability:
            logger.info(
                f"Source {source_id}: Whisper guessed '{detected}' at {probability:.2f}, below the "
                f"{rules.min_detection_probability:.2f} needed to reject it; transcribing as {pack.name}."
            )
            return

        raise LanguageMismatchError(
            f"Source {source_id} is spoken in '{detected}' (probability {probability:.2f}), but this appliance "
            f"transcribes {pack.name} ('{rules.whisper_language}'). The file was not transcribed: forcing "
            f"{pack.name} on it would produce text nobody said."
        )

    def get_transcript(self, source_id: uuid.UUID) -> Optional[Transcript]:
        return self._repo.get_by_source_id(source_id)

    def update_transcript(self, transcript: Transcript) -> Transcript:
        return self._repo.update(transcript)


def _normalize_segment_script(segment: TranscriptSegment, script: str) -> TranscriptSegment:
    return segment.model_copy(update={
        "text": normalize_script(segment.text, script),
        "words": [word.model_copy(update={"word": normalize_script(word.word, script)}) for word in segment.words],
    })
