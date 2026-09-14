# Canvas: extraction_api_v4
# Path: app/features/frame_extraction/service/extraction_api.py

import os
import uuid
import tempfile
import concurrent.futures
from pathlib import Path
import logging
from typing import List

from app.core.config import settings
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames, ExtractedFrame
from app.features.frame_extraction.domain.interfaces import IFFmpegExtractionAdapter, IFrameExtractionRepository
from app.features.frame_extraction.utils.quality_filter import QualityFilter
from app.features.frame_extraction.utils.frame_deduplicator import FrameDeduplicator
from app.features.frame_extraction.utils.frame_selector import select_diverse_frames
from app.features.frame_extraction.utils.error_mapper import map_extraction_error

from app.features.storage.service.storage_api import StorageAPI

logger = logging.getLogger(__name__)


class FrameExtractionAPI:
    """
    Orchestrates the strictly CPU-bound extraction of visual frames for a given text chunk.
    Filters out blurry/duplicate frames and caps each chunk at its most visually distinct frames
    before moving the survivors to the Storage Vault.
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

    def process_chunks_batch(
            self,
            source_id: uuid.UUID,
            video_path: str,
            chunks: List[ContextualizedChunk]
    ) -> List[ExtractedChunkFrames]:
        """
        Processes multiple chunks concurrently. Bypasses the GIL limitations by relying
        on OS-level FFmpeg subprocesses and OpenCV C++ bindings.
        """
        results: List[ExtractedChunkFrames] = []

        if not chunks:
            return results

        # Each worker drives one multi-threaded FFmpeg decode, so half the cores avoids oversubscription
        max_workers = settings.FRAME_EXTRACTION_WORKERS or max(2, (os.cpu_count() or 4) // 2)
        logger.info(f"Spinning up {max_workers} concurrent CPU workers for Frame Extraction...")

        def _worker(target_chunk: ContextualizedChunk) -> ExtractedChunkFrames:
            return self.process_chunk(source_id, video_path, target_chunk)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk_map = {executor.submit(_worker, c): c for c in chunks}

            for future in concurrent.futures.as_completed(future_to_chunk_map):
                completed_chunk = future_to_chunk_map[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # If a single thread fails, bubble the error up so the Job Manager
                    # correctly marks the overarching Job as FAILED and saves the stack trace.
                    raise map_extraction_error(e,
                                               context=f"Parallel batch processing for chunk {completed_chunk.chunk_id}") from e

        return results

    def process_chunk(
            self,
            source_id: uuid.UUID,
            video_path: str,
            chunk: ContextualizedChunk
    ) -> ExtractedChunkFrames:
        """Isolated processing logic for a single chunk."""
        try:
            # 1. Idempotency Check
            existing_record = self._repo.get_by_chunk_id(chunk.chunk_id)
            if existing_record:
                logger.info(f"Frames already extracted for chunk {chunk.chunk_id}. Skipping.")
                return existing_record

            # 2. Secure Temporary Directory Context (frames that are not vaulted are removed with it)
            with tempfile.TemporaryDirectory(prefix=f"onyx_chunk_{chunk.chunk_id}_") as tmpdir:
                tmp_path = Path(tmpdir)

                # 3. Step 1: Single-pass extraction of the chunk's interval (1 FPS)
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

                # 6. Step 4: Keep only the most visually distinct frames. VLM cost scales with frame count,
                # and a long chunk otherwise sends every changed second to the GPU.
                selected_frames, dropped_frames = select_diverse_frames(unique_frames, settings.FRAME_MAX_PER_CHUNK)
                if dropped_frames:
                    logger.info(
                        f"Chunk {chunk.chunk_id}: kept {len(selected_frames)} of {len(unique_frames)} distinct frames."
                    )

                # 7. Step 5: Vault Storage & Domain Mapping
                final_extracted_frames: List[ExtractedFrame] = []

                for timestamp_sec, img_path, quality_score, phash_val in selected_frames:
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

                # 8. Aggregate and Persist Mapping
                chunk_frames = ExtractedChunkFrames(
                    source_id=source_id,
                    chunk_id=chunk.chunk_id,
                    frames=final_extracted_frames
                )

                return self._repo.save(chunk_frames)

        except Exception as e:
            raise map_extraction_error(e, context=f"FrameExtractionAPI.process_chunk for {chunk.chunk_id}") from e

    def _save_empty_result(self, source_id: uuid.UUID, chunk: ContextualizedChunk) -> ExtractedChunkFrames:
        empty_frames = ExtractedChunkFrames(
            source_id=source_id,
            chunk_id=chunk.chunk_id,
            frames=[]
        )
        return self._repo.save(empty_frames)
