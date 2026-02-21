import os
import tempfile
import uuid
from pathlib import Path

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
        """
        Initializes the Audio Extractor Service.
        Injects the data layer adapters and the external StorageAPI dependency.
        """
        self._ffmpeg = ffmpeg_adapter
        self._repo = repository
        self._storage = storage_api

    def extract_audio_from_video(self, video_source_id: uuid.UUID) -> AudioExtraction:
        # 1. Fetch and validate the original video source from Storage
        video_source = self._storage.get_source(video_source_id)

        if not video_source:
            raise AudioExtractorError(f"Cannot extract audio: Source ID {video_source_id} not found.")

        if video_source.source_type != SourceType.VIDEO:
            raise AudioExtractorError(f"Cannot extract audio: Source {video_source_id} is not a VIDEO.")

        if not video_source.file:
            raise AudioExtractorError(
                f"Cannot extract audio: Video Source {video_source_id} has no physical file attached.")

        original_video_path = video_source.file.file_path
        temp_path = None

        try:
            # 2. Extract Audio (Adapter handles all OS temp file logic)
            temp_path = self._ffmpeg.extract_audio(original_video_path)

            # 3. Vault the extracted file securely
            # (delete_original=True automatically cleans up the adapter's temp file upon success)
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
            # Fallback Cleanup: In case the Storage API or DB crashes AFTER FFmpeg succeeds
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise e