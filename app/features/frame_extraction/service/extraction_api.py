# Canvas: extraction_api_v1
# Path: app/features/frame_extraction/service/extraction_api.py

import uuid
import tempfile
from pathlib import Path
import logging
from typing import List

from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames, ExtractedFrame
from app.features.frame_extraction.domain.interfaces import IFFmpegExtractionAdapter, IFrameExtractionRepository
from app.features.frame_extraction.utils.quality_filter import QualityFilter
from app.features.frame_extraction.utils.frame_deduplicator import FrameDeduplicator
from app.features.frame_extraction.utils.error_mapper import map_extraction_error

from app.features.storage.service.storage_api import StorageAPI

logger = logging.getLogger(__name__)


class FrameExtractionAPI:
    """
    Orchestrates the strictly CPU-bound extraction of visual frames for a given text chunk.
    Filters out blurry/duplicate frames before moving the survivors to the Storage Vault.
    """

    def __init__(
            self,
            repository: IFrameExtractionRepository,
            ffmpeg_adapter: IFFmpegExtractionAdapter,
            storage_api: StorageAPI,
            quality_filter: QualityFilter,
            deduplicator: FrameDeduplicator
    ):
        self._repo = repository
        self._ffmpeg = ffmpeg_adapter
        self._storage = storage_api
        self._quality_filter = quality_filter
        self._deduplicator = deduplicator

    def process_chunk(
            self,
            source_id: uuid.UUID,
            video_path: str,
            chunk: ContextualizedChunk
    ) -> ExtractedChunkFrames:

        try:
            # 1. Idempotency Check: Don't re-extract if we already survived a crash here
            existing_record = self._repo.get_by_chunk_id(chunk.chunk_id)
            if existing_record:
                logger.info(f"Frames already extracted for chunk {chunk.chunk_id}. Skipping.")
                return existing_record

            # 2. Secure Temporary Directory Context
            with tempfile.TemporaryDirectory(prefix=f"onyx_chunk_{chunk.chunk_id}_") as tmpdir:
                tmp_path = Path(tmpdir)

                # 3. Step 1: Raw Fast-Seek Extraction (1 FPS)
                raw_frames = self._ffmpeg.extract_raw_frames(
                    video_path=video_path,
                    output_dir=tmp_path,
                    chunk=chunk,
                    fps=1
                )

                if not raw_frames:
                    return self._save_empty_result(source_id, chunk)

                # 4. Step 2: CPU Quality Filtering (Blur/Motion rejection)
                high_quality_frames = self._quality_filter.filter_frames(raw_frames)

                if not high_quality_frames:
                    return self._save_empty_result(source_id, chunk)

                # 5. Step 3: CPU pHash Deduplication (Static Scene collapse)
                unique_frames = self._deduplicator.deduplicate(high_quality_frames)

                # 6. Step 4: Vault Storage & Domain Mapping
                final_extracted_frames: List[ExtractedFrame] = []

                for timestamp_sec, img_path, quality_score, phash_val in unique_frames:
                    # Move from temp dir to the highly secure Onyx Storage vault
                    vaulted_file = self._storage.ingest_raw_file(
                        file_path=str(img_path),
                        delete_original=True
                    )

                    final_extracted_frames.append(
                        ExtractedFrame(
                            file_id=vaulted_file.id,
                            timestamp_sec=timestamp_sec,
                            phash_value=phash_val,
                            quality_score=quality_score
                        )
                    )

                # 7. Aggregate and Persist Mapping
                chunk_frames = ExtractedChunkFrames(
                    source_id=source_id,
                    chunk_id=chunk.chunk_id,
                    frames=final_extracted_frames
                )

                return self._repo.save(chunk_frames)

        except Exception as e:
            raise map_extraction_error(e, context=f"FrameExtractionAPI.process_chunk for {chunk.chunk_id}") from e

    def _save_empty_result(self, source_id: uuid.UUID, chunk: ContextualizedChunk) -> ExtractedChunkFrames:
        """Saves an empty record if all frames were dropped (e.g., completely blurry chunk)."""
        empty_frames = ExtractedChunkFrames(
            source_id=source_id,
            chunk_id=chunk.chunk_id,
            frames=[]
        )
        return self._repo.save(empty_frames)