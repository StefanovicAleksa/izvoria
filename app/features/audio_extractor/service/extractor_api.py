# Canvas: extractor_api_v4
# Path: app/features/audio_extractor/service/extractor_api.py
import uuid

from app.features.audio_extractor.domain.entities import AudioExtraction
from app.features.audio_extractor.domain.interfaces import IFFmpegAdapter, IAudioExtractionRepository
from app.features.audio_extractor.exceptions.base import AudioExtractorError

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import SourceType

# Every audio-bearing source is normalized to one 16kHz mono PCM WAV that all downstream audio
# phases (VAD, domain adaptation, transcription, diarization) read, so they decode identical samples.
_EXTRACTABLE_SOURCE_TYPES = {SourceType.VIDEO, SourceType.AUDIO}


class AudioExtractorAPI:
    def __init__(
        self,
        ffmpeg_adapter: IFFmpegAdapter,
        repository: IAudioExtractionRepository,
        storage_api: StorageAPI
    ):
        self._ffmpeg = ffmpeg_adapter
        self._repo = repository
        self._storage = storage_api

    def extract_audio_from_video(self, video_source_id: uuid.UUID) -> AudioExtraction:
        """
        Extracts (or, for audio sources, normalizes) the source's audio to a vaulted 16kHz mono WAV.
        The name is kept for compatibility; VIDEO and AUDIO sources are both accepted.
        """
        # --- NEW: Idempotency Check ---
        # Instantly bypass FFmpeg if we already extracted audio for this source
        existing_extraction = self._repo.get_latest_by_source_id(video_source_id)
        if existing_extraction:
            return existing_extraction
        # ------------------------------

        # 1. Fetch and validate the original source from Storage
        source = self._storage.get_source(video_source_id)

        if not source:
            raise AudioExtractorError(f"Cannot extract audio: Source ID {video_source_id} not found.")

        if source.source_type not in _EXTRACTABLE_SOURCE_TYPES:
            raise AudioExtractorError(f"Cannot extract audio: Source {video_source_id} is not a VIDEO or AUDIO source.")

        if not source.file:
            raise AudioExtractorError(f"Cannot extract audio: Source {video_source_id} has no physical file attached.")

        original_media_path = source.file.file_path
        temp_path = None

        try:
            # 2. Extract Audio (Adapter handles all OS temp file logic)
            temp_path = self._ffmpeg.extract_audio(original_media_path)

            # 3. Vault the extracted file securely
            audio_file = self._storage.ingest_raw_file(
                file_path=str(temp_path),
                delete_original=True
            )

            # 4. Record Lineage
            extraction_record = AudioExtraction(
                id=uuid.uuid4(),
                source_id=source.id,
                extracted_file_id=audio_file.id
            )

            return self._repo.create(extraction_record)

        except Exception as e:
            # Fallback Cleanup
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise e

    def get_extracted_audio_path(self, source_id: uuid.UUID) -> str:
        """
        Returns the vaulted path of the source's normalized 16kHz mono WAV.
        Downstream phases must read this file, never the original media: decoding the original
        (e.g. 48kHz stereo AAC inside an MP4) again in every phase wastes minutes and RAM and can
        shift timestamps relative to the VAD map computed from the WAV.
        """
        extraction = self._repo.get_latest_by_source_id(source_id)
        if not extraction:
            raise AudioExtractorError(f"No extracted audio recorded for Source {source_id}. Run audio extraction first.")

        return str(self._storage.get_file_path(extraction.extracted_file_id))
