# Canvas: extractor_api_v3
# Path: app/features/audio_extractor/service/extractor_api.py
import uuid

from app.features.audio_extractor.domain.entities import AudioExtraction
from app.features.audio_extractor.domain.interfaces import IFFmpegAdapter, IAudioExtractionRepository
from app.features.audio_extractor.exceptions.base import AudioExtractorError

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import SourceType

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
        # --- NEW: Idempotency Check ---
        # Instantly bypass FFmpeg if we already extracted audio for this video source
        existing_extraction = self._repo.get_latest_by_source_id(video_source_id)
        if existing_extraction:
            return existing_extraction
        # ------------------------------

        # 1. Fetch and validate the original video source from Storage
        video_source = self._storage.get_source(video_source_id)

        if not video_source:
            raise AudioExtractorError(f"Cannot extract audio: Source ID {video_source_id} not found.")

        if video_source.source_type != SourceType.VIDEO:
            raise AudioExtractorError(f"Cannot extract audio: Source {video_source_id} is not a VIDEO.")

        if not video_source.file:
            raise AudioExtractorError(f"Cannot extract audio: Video Source {video_source_id} has no physical file attached.")

        original_video_path = video_source.file.file_path
        temp_path = None

        try:
            # 2. Extract Audio (Adapter handles all OS temp file logic)
            temp_path = self._ffmpeg.extract_audio(original_video_path)

            # 3. Vault the extracted file securely
            audio_file = self._storage.ingest_raw_file(
                file_path=str(temp_path),
                delete_original=True
            )

            # 4. Record Lineage
            extraction_record = AudioExtraction(
                id=uuid.uuid4(),
                source_id=video_source.id,
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