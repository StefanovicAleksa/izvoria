"""
Makes each uploaded video's 720p copy on the VPS, before the GPU box ever sees it.

A firm's raw footage can run to ~100 GB an hour, and the box needs none of that resolution: the vault keeps a 720p
H.264/AAC copy anyway (docs/web_deploy.md, "Storage"), so the copy is made here, first, and only the copy travels to
the box and is ingested. Transfer to the rented GPU (which its host bills per byte) shrinks by one to two orders of
magnitude, and the box's compaction, seeing an already-compact MP4, leaves it alone.

A video batch is queued as `preparing` (PREPARE_VIDEOS=true); the box only claims `queued` jobs. Per video:
1. ffprobe. An MP4 that is already compact (the box's own rule, `is_compact_video`) is left as it is.
2. The original is hashed (sha256, as ingestion hashes an upload), so the same upload still finds its source.
3. ffmpeg, niced, into a hidden temporary file beside it (the same arguments as the box's compaction, with
   PREPARE_PRESET and a speech-safe audio bitrate). A copy whose length differs from the original's is refused and
   the original kept: the box then ingests and compacts it as before.
4. The upload record (app/features/storage/utils/upload_record.py) gets the copy's name, the original's hash and
   name; then the copy takes its place and the original is deleted.

A batch survives a gateway restart: a temporary file is removed, a copy already recorded is kept, and the rest is
done again. A cancelled job stops after the file in hand. Runs in the gateway's process (one job at a time, one
file at a time; ffmpeg itself uses every core), which then wakes the box.
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from app.features.storage.utils.upload_record import UploadRecord, read_all, write_record
from deploy.instance.agent.uploads import VIDEO_SUFFIXES
from web_deploy.common.uploads import batch_files
from web_deploy.common.video_copy import CopySettings, Runner, copy_name, make_copy
from web_deploy.ingest_worker.queue import IngestJob, IngestQueue, JobStatus

logger = logging.getLogger(__name__)

TEMP_PREFIX = ".preparing-"


@dataclass(frozen=True)
class PrepareSettings:
    enabled: bool = False
    copy: CopySettings = field(default_factory=CopySettings)

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "PrepareSettings":
        env = dict(os.environ if env is None else env)
        return cls(
            enabled=env.get("PREPARE_VIDEOS", "false").strip().lower() in ("1", "true", "yes"),
            copy=CopySettings(
                max_height=int(env.get("PREPARE_VIDEO_MAX_HEIGHT", "720")),
                crf=int(env.get("PREPARE_VIDEO_CRF", "26")),
                preset=env.get("PREPARE_PRESET", "faster").strip() or "faster",
                audio_bitrate=env.get("PREPARE_AUDIO_BITRATE", "128k").strip() or "128k",
            ),
        )


class VideoPreparer:
    def __init__(self, queue: IngestQueue, settings: PrepareSettings, run: Runner = subprocess.run) -> None:
        self._queue = queue
        self._settings = settings
        self._run = run

    def run_once(self) -> Optional[IngestJob]:
        """Prepares the oldest job waiting for it. Returns it once it is handed to the box's line, else None."""
        job = self._queue.next_to_prepare()
        if job is None:
            return None
        folder = Path(job.batch_dir)
        try:
            self.prepare_folder(job, folder)
        except _Cancelled:
            logger.info(f"Job {job.id} was cancelled while its videos were being prepared")
            return None
        files = len(batch_files(folder))
        return job if self._queue.mark_prepared(job.id, files or None) else None

    def prepare_folder(self, job: IngestJob, folder: Path) -> Dict[str, UploadRecord]:
        records = self._tidy(folder)
        videos = [p for p in batch_files(folder)
                  if p.suffix.lower() in VIDEO_SUFFIXES and p.name not in records]
        total = len(videos) + len(records)
        for index, video in enumerate(videos):
            current = self._queue.get(job.id)
            if current is None or current.status is not JobStatus.PREPARING:
                raise _Cancelled()
            self._progress(job.id, len(records) + index, total, video.name)
            try:
                prepared = self.prepare_file(video, set(p.name for p in folder.iterdir()))
            except Exception as exc:
                # The original stays: the box ingests and compacts it exactly as before this step existed
                logger.warning(f"Kept {video.name} as uploaded; its 720p copy failed: {exc}")
                continue
            if prepared is None:
                continue
            name, record, copy = prepared
            records[name] = record
            # Recorded before the copy takes its place, so a restart in between finishes the swap (_tidy)
            write_record(folder, records)
            copy.replace(folder / name)
            if video.name != name:
                video.unlink(missing_ok=True)
        self._progress(job.id, total, total, None)
        return records

    def prepare_file(self, source: Path, taken: set):
        """(the copy's name, its record, the temporary copy), or None when the original is kept as it is."""
        temp = source.with_name(f"{TEMP_PREFIX}{source.stem}.mp4")
        record = make_copy(source, temp, self._settings.copy, self._run)
        if record is None:
            return None
        return copy_name(source, taken), record, temp

    def _tidy(self, folder: Path) -> Dict[str, UploadRecord]:
        """After a restart: temporary files go, recorded copies stay (their originals are deleted if still there)."""
        for leftover in folder.glob(f"{TEMP_PREFIX}*"):
            leftover.unlink(missing_ok=True)
        records = {name: record for name, record in read_all(folder).items() if (folder / name).is_file()}
        for name, record in records.items():
            if record.original_name != name:
                (folder / record.original_name).unlink(missing_ok=True)
        return records

    def _progress(self, job_id: str, done: int, total: int, detail: Optional[str]) -> None:
        try:
            self._queue.record_progress(job_id, {"phase": "preparing", "state": "started", "detail": detail,
                                                 "done": done, "total": total})
        except Exception as exc:
            logger.debug(f"Could not record progress: {exc}")


class _Cancelled(Exception):
    pass
