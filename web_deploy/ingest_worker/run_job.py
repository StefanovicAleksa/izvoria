"""
One ingestion job, in a child process the supervisor started with the firm's settings in its environment.

It builds the appliance's own engine (the desktop's, unchanged: in-process llama.cpp, one model at a time) and runs
the batch through the same pipelines, writing progress to the job row so the firm's users can watch the wait list.

Run: python -m web_deploy.ingest_worker.run_job <job id>
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set

from web_deploy.common import youtube
from web_deploy.ingest_worker.queue import IngestMode, IngestQueue

logger = logging.getLogger(__name__)

# Progress is written to the database at most this often; a tick per file per phase would hammer it
PROGRESS_INTERVAL_SECONDS = 2.0


class ProgressWriter:
    """
    Also records what the web app needs to draw the desktop's per-step bars: how many units of work each engine
    job has finished (`done_by_phase`) and how many distinct files the batch has shown so far (`files`). The web
    app folds jobs into steps itself, as the desktop's progress_stages.py does.
    """

    def __init__(self, queue: IngestQueue, job_id: str, clock=time.monotonic, wall=time.time) -> None:
        self._queue = queue
        self._job_id = job_id
        self._clock = clock
        self._wall = wall
        self._last = float("-inf")
        self._completed = 0
        self._failed = 0
        self._done_by_phase: Dict[str, int] = {}
        self._files: Set[str] = set()

    def __call__(self, phase: str, state: str, source_id=None, detail: Optional[str] = None) -> None:
        phase, state = str(phase), str(state)
        if state == "completed":
            self._completed += 1
        elif state == "failed":
            self._failed += 1
        if source_id is not None:
            self._files.add(str(source_id))
        # The first tick of a job is always written, so a step change never waits for the next completion
        new_phase = phase not in self._done_by_phase
        if state in ("completed", "failed") or new_phase:
            self._done_by_phase[phase] = self._done_by_phase.get(phase, 0) + (state != "started")
        now = self._clock()
        if now - self._last < PROGRESS_INTERVAL_SECONDS and state == "started" and not new_phase:
            return
        self._last = now
        try:
            self._queue.record_progress(self._job_id, {
                "phase": phase, "state": state, "detail": detail,
                "steps_completed": self._completed, "steps_failed": self._failed,
                "done_by_phase": dict(self._done_by_phase), "files": len(self._files),
                # Wall time of this tick: the web app says "still working" when a step has been quiet a while
                "updated_at": self._wall(),
            })
        except Exception as exc:
            logger.debug(f"Could not record progress: {exc}")


def _download_errors(batch_dir: str) -> dict:
    """A YouTube batch's videos that could not be downloaded, listed with the files that could not be ingested."""
    manifest = youtube.read_manifest(Path(batch_dir))
    return dict(manifest.get("errors", {})) if manifest else {}


def run(job_id: str, queue: IngestQueue) -> int:
    job = queue.get(job_id)
    if job is None:
        logger.error(f"Job {job_id} does not exist.")
        return 2

    from app.core.config import settings
    from app.engine.service.bootstrap import IzvoriaBootstrapper
    from web_deploy.api.firm_db import ensure_firm_database

    engine = None
    try:
        # A seeded demo firm can be ingested into before anyone has signed in to it
        ensure_firm_database(settings.POSTGRES_DB, settings.POSTGRES_SERVER, settings.POSTGRES_PORT,
                             settings.POSTGRES_USER, settings.POSTGRES_PASSWORD,
                             os.environ.get("POSTGRES_ADMIN_DB", "postgres"))
        engine = IzvoriaBootstrapper.build(progress_listener=ProgressWriter(queue, job_id))
        engine.start()
        ingest = {
            IngestMode.VIDEO: engine.ingest_videos,
            IngestMode.AUDIO: engine.ingest_audios,
            IngestMode.DOCUMENT: engine.ingest_documents,
            IngestMode.IMAGE: engine.ingest_images,
        }[job.mode]
        report = ingest(job.batch_dir)
        queue.finish(job_id, {
            "scanned": report.total_files_scanned,
            "succeeded": report.successful_ingestions,
            "failed": report.failed_ingestions,
            "vectors_created": report.vectors_created,
            "clips_extracted": report.clips_extracted,
            "pages_converted": getattr(report, "pages_converted", 0),
            "images_described": getattr(report, "images_described", 0),
            "storage_bytes_saved": getattr(report, "storage_bytes_saved", 0),
            "duration_seconds": report.total_duration_seconds,
            "errors": {**_download_errors(job.batch_dir),
                       **{os.path.basename(k): v for k, v in sorted(report.errors.items())}},
        })
        return 0
    except Exception as exc:
        logger.exception(f"Ingestion job {job_id} failed")
        queue.fail(job_id, str(exc) or type(exc).__name__)
        return 1
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from web_deploy.api.config import ApiSettings
    queue = IngestQueue.connect(ApiSettings.from_env().control_database_url)
    sys.exit(run(sys.argv[1], queue))


if __name__ == "__main__":
    main()
