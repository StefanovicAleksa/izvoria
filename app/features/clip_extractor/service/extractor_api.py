# Canvas: clip_extractor_api_v2
# Path: app/features/clip_extractor/service/extractor_api.py

import uuid
import logging
from typing import List

from app.features.context_unification.domain.entities import UnifiedTranscript
from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import SourceType

from app.features.clip_extractor.domain.interfaces import IFFmpegVideoAdapter, IVideoClipRepository
from app.features.clip_extractor.domain.entities import VideoClip
from app.features.clip_extractor.exceptions.base import ClipExtractorError
from app.features.clip_extractor.utils.error_mapper import map_clip_extractor_error

logger = logging.getLogger(__name__)


class ClipExtractorAPI:
    def __init__(
            self,
            ffmpeg_adapter: IFFmpegVideoAdapter,
            repository: IVideoClipRepository,
            storage_api: StorageAPI
    ):
        self._ffmpeg = ffmpeg_adapter
        self._repo = repository
        self._storage = storage_api

    def extract_clips_from_transcript(self, transcript: UnifiedTranscript) -> List[VideoClip]:
        source_id = transcript.source_id

        # 1. Idempotency Check: Don't extract if we already did it for this source
        existing_clips = self._repo.get_by_source_id(source_id)
        if existing_clips:
            logger.info(f"Clips already exist for Source {source_id}. Skipping extraction.")
            return existing_clips

        # Guard Clause: Ensure there are chunks to process
        if not transcript.chunks:
            return []

        # 2. Fetch the original Video Source from Storage
        video_source = self._storage.get_source(source_id)
        if not video_source:
            raise ClipExtractorError(f"Cannot extract clips: Source {source_id} not found in storage.")

        if video_source.source_type != SourceType.VIDEO:
            raise ClipExtractorError(f"Cannot extract clips: Source {source_id} is not a VIDEO.")

        if not video_source.file:
            raise ClipExtractorError(f"Cannot extract clips: Source {source_id} has no physical file attached.")

        original_video_path = video_source.file.file_path
        new_video_clips: List[VideoClip] = []

        try:
            # 3. Loop through Unification Chunks and Extract
            for chunk in transcript.chunks:
                # Add a small buffer to the clip (e.g., 0.5s) to ensure the visual context isn't abruptly cut
                safe_start = max(0.0, chunk.start_sec - 0.5)
                safe_end = chunk.end_sec + 0.5

                temp_clip_path = None
                try:
                    # Execute FFmpeg Subprocess
                    temp_clip_path = self._ffmpeg.extract_clip(
                        source_path=original_video_path,
                        start_sec=safe_start,
                        end_sec=safe_end
                    )

                    # Vault the physical file securely
                    vaulted_file = self._storage.ingest_raw_file(
                        file_path=str(temp_clip_path),
                        delete_original=True  # Storage API will consume and delete our temp file
                    )

                    # Construct Domain Record
                    new_video_clips.append(
                        VideoClip(
                            id=uuid.uuid4(),
                            source_id=source_id,
                            chunk_id=chunk.chunk_id,
                            extracted_file_id=vaulted_file.id,
                            start_sec=safe_start,
                            end_sec=safe_end
                        )
                    )
                except Exception as loop_err:
                    # Clean up the specific temp file if it crashed before storage ingestion
                    if temp_clip_path and temp_clip_path.exists():
                        try:
                            temp_clip_path.unlink()
                        except OSError:
                            pass
                    logger.error(f"Failed to extract chunk {chunk.chunk_id}: {str(loop_err)}")
                    raise loop_err

            # 4. Bulk Save to PostgreSQL
            if new_video_clips:
                self._repo.create_bulk(new_video_clips)
                logger.info(f"Successfully extracted and vaulted {len(new_video_clips)} clips for Source {source_id}.")

            return new_video_clips

        except Exception as e:
            # Clips vaulted earlier in this run have no DB rows yet. Remove them, otherwise they
            # become orphaned files and the idempotency check re-extracts them on the next run.
            self._discard_vaulted_clips(new_video_clips)
            raise map_clip_extractor_error(e, context=f"ClipExtractorAPI for Source {source_id}") from e

    def discard_clips(self, source_id: uuid.UUID, dry_run: bool = False) -> int:
        """
        Deletes every clip of a source, returning the bytes freed. A citation of one of its chunks then points at
        the original recording (with its start and end) instead of a clip.
        """
        freed = 0
        for clip in self._repo.get_by_source_id(source_id):
            record = self._storage.get_file(clip.extracted_file_id)
            freed += record.file_size_bytes if record else 0
            if not dry_run:
                # Deleting the file record cascades the video_clips row away
                self._storage.delete_raw_file(clip.extracted_file_id)
        return freed

    def _discard_vaulted_clips(self, clips: List[VideoClip]) -> None:
        for clip in clips:
            try:
                self._storage.delete_raw_file(clip.extracted_file_id)
            except Exception as cleanup_err:
                logger.error(f"Failed to remove orphaned clip file {clip.extracted_file_id}: {cleanup_err}")
