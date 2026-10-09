"""
Runs the ingestion wait list, one batch at a time, each in a fresh child process.

A child per job, not one long-lived worker, because:
- the torchcodec guard leaks between retrieval and ingestion inside one process (CLAUDE.md, Known gotchas);
- a child that exits returns every byte of VRAM it touched, llama.cpp fragmentation included, to the live side;
- each job runs in its own firm's database and vault and in its batch's language, which are process-wide settings
  for the appliance's ingestion code (`settings.use_language`), so a fresh process per job is the clean way to set
  them. Starting torch costs ~10 s, nothing next to an ingestion.

The child runs the appliance exactly as the desktop does (LLM_BACKEND=llama_cpp, single-slot pool), but sees the card
minus VRAM_HEADROOM_MB, so it never takes the room the live models need to answer.

Uploads are deleted once their job is over (done, failed or cancelled): ingestion copies every file into the firm's
vault, so the upload folder would otherwise keep a second copy of everything forever. A job re-queued by a stop keeps
its folder, since the retry reads it again. A periodic sweep removes folders no job will read any more (cancelled jobs,
uploads never queued) once nothing has been written to them for a day.

Run: python -m web_deploy.ingest_worker.supervisor  (supervisord, see web_deploy/provision/supervisord.conf)
"""

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from web_deploy.common import youtube
from web_deploy.common.tenancy import database_name, firm_root
from web_deploy.ingest_worker.gpu_arbiter import GpuArbiter, arbiter_from_env
from web_deploy.ingest_worker.queue import IngestJob, IngestQueue, JobOrigin

logger = logging.getLogger(__name__)

POLL_SECONDS = 5.0
# An upload folder no queued or running job refers to is removed once untouched this long: an upload still arriving
# (its .part file grows) or a batch about to be queued is always newer
UPLOAD_SWEEP_AGE_SECONDS = 24 * 3600
UPLOAD_SWEEP_INTERVAL_SECONDS = 3600


def child_environment(job: IngestJob, tenants_root: Path, base: Dict[str, str]) -> Dict[str, str]:
    """The appliance's settings for one job: the firm's database and directories, the batch's language."""
    env = firm_environment(job.firm, job.language, tenants_root, base)
    env["WEB_INGEST_JOB_ID"] = job.id
    return env


def firm_environment(firm: str, language: str, tenants_root: Path, base: Dict[str, str]) -> Dict[str, str]:
    """The appliance's settings for engine work on one firm's data, run in-process as the desktop runs it."""
    root = firm_root(tenants_root, firm)
    env = dict(base)
    env.update({
        "POSTGRES_DB": database_name(firm),
        "ONYX_STORAGE_PATH": str(root / "vault"),
        "REPORTS_PATH": str(root / "reports"),
        "APPLIANCE_LANGUAGE": language,
        "REMEMBER_LANGUAGE": "false",
        # Ingestion runs in-process on the ingestion half of the card, exactly as on the appliance
        "LLM_BACKEND": "llama_cpp",
        "MODEL_KEEP_WARM": "false",
        "MODEL_PINNED_TYPES": "[]",
        "MODEL_EXTERNAL_TYPES": "[]",
        "ENGINE_CONCURRENT_LIVE": "false",
        "VRAM_HEADROOM_MB": base.get("INGEST_VRAM_HEADROOM_MB", "1500"),
    })
    return env


class Supervisor:
    def __init__(self, queue: IngestQueue, tenants_root: Path,
                 spawn: Optional[Callable[[IngestJob, Dict[str, str]], int]] = None,
                 base_env: Optional[Dict[str, str]] = None, arbiter: Optional[GpuArbiter] = None,
                 fetch: Optional[Callable[[IngestJob, Dict[str, str]], int]] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._queue = queue
        self._arbiter = arbiter or GpuArbiter()
        self._tenants_root = tenants_root
        self._spawn = spawn or self._spawn_child
        self._fetch = fetch or self._spawn_fetch
        self._base_env = dict(base_env if base_env is not None else os.environ)
        self._child: Optional[subprocess.Popen] = None
        self._stopping = False
        self._clock = clock
        self._last_sweep = float("-inf")

    def run_once(self) -> bool:
        """Runs the next job if there is one. Returns whether it ran one."""
        job = self._queue.claim_next()
        if job is None:
            self._arbiter.when_idle()
            if self._clock() - self._last_sweep >= UPLOAD_SWEEP_INTERVAL_SECONDS:
                self.sweep_uploads()
            return False
        try:
            self._run_job(job)
        finally:
            self._remove_finished_upload(job)
        return True

    def _run_job(self, job: IngestJob) -> None:
        if job.origin is JobOrigin.YOUTUBE and not self._download(job):
            return
        self._queue.record_progress(job.id, {"phase": "waiting_for_gpu", "state": "started", "detail": None})
        self._arbiter.before_job()
        logger.info(f"Ingesting job {job.id} for {job.firm}: {job.file_count} {job.mode.value} file(s) in {job.language}")
        try:
            code = self._spawn(job, child_environment(job, self._tenants_root, self._base_env))
        except Exception as exc:
            logger.exception(f"Could not start the ingestion child for {job.id}")
            self._queue.fail(job.id, f"The ingestion worker could not start: {exc}")
            return
        # The child records done/failed itself; an exit without a verdict is a crash (an OOM kill, a segfault)
        final = self._queue.get(job.id)
        if final is not None and final.status.value == "running":
            self._queue.fail(job.id, f"The ingestion worker stopped unexpectedly (exit code {code}).")

    def _remove_finished_upload(self, job: IngestJob) -> None:
        """The batch's upload folder, once its job can never run again. Never during a stop: the job is re-queued."""
        if self._stopping:
            return
        try:
            final = self._queue.get(job.id)
            if final is None or final.status.value not in ("done", "failed", "cancelled"):
                return
            folder = self._upload_folder(job.firm, Path(job.batch_dir))
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)
                logger.info(f"Removed the uploads of job {job.id}: their files are in the vault")
        except Exception as exc:
            # The sweep catches it later; a failed cleanup must not stop the wait list
            logger.warning(f"Could not remove the uploads of job {job.id}: {exc}")

    def sweep_uploads(self) -> int:
        """Removes upload folders no queued or running job refers to and nothing has touched for a day."""
        self._last_sweep = self._clock()
        try:
            stale = stale_upload_folders(self._queue, self._tenants_root, self._clock())
        except Exception as exc:
            logger.warning(f"Upload sweep skipped: could not read the queue ({exc})")
            return 0
        for folder in stale:
            shutil.rmtree(folder, ignore_errors=True)
        if stale:
            logger.info(f"Upload sweep removed {len(stale)} folder(s) no job will read")
        return len(stale)

    def _upload_folder(self, firm: str, batch_dir: Path) -> Optional[Path]:
        """`batch_dir` if it really is a batch under that firm's uploads, else None (never delete anything else)."""
        try:
            uploads = (firm_root(self._tenants_root, firm) / "uploads").resolve()
            folder = batch_dir.resolve()
        except Exception:
            return None
        if folder.parent.parent != uploads or not folder.is_dir():
            logger.warning(f"Not removing {batch_dir}: it is not a batch folder under {uploads}")
            return None
        return folder

    def _download(self, job: IngestJob) -> bool:
        """
        Fetches a YouTube batch's videos, before the GPU is asked for: on dev hardware chat keeps answering while they
        download. Returns whether the job should go on to be ingested.
        """
        if not youtube.pending(Path(job.batch_dir)):
            return True  # downloaded before a restart interrupted the ingestion
        logger.info(f"Downloading the videos of job {job.id} for {job.firm}")
        try:
            code = self._fetch(job, dict(self._base_env))
        except Exception as exc:
            logger.exception(f"Could not start the download for {job.id}")
            self._queue.fail(job.id, f"The video download could not start: {exc}")
            return False
        if self._stopping:
            return False  # re-queued at the next start (recover_interrupted); the partial files resume
        final = self._queue.get(job.id)
        if final is None or final.status.value != "running":
            return False  # failed by the download itself: nothing could be fetched
        if code != 0:
            self._queue.fail(job.id, f"The video download stopped unexpectedly (exit code {code}).")
            return False
        return True

    def _spawn_fetch(self, job: IngestJob, env: Dict[str, str]) -> int:
        self._child = subprocess.Popen(
            [sys.executable, "-m", "web_deploy.ingest_worker.fetch_links", job.id],
            env=env, cwd=env.get("ONYX_ROOT") or None,
        )
        try:
            return self._child.wait()
        finally:
            self._child = None

    def _spawn_child(self, job: IngestJob, env: Dict[str, str]) -> int:
        self._child = subprocess.Popen(
            [sys.executable, "-m", "web_deploy.ingest_worker.run_job", job.id],
            env=env, cwd=env.get("ONYX_ROOT") or None,
        )
        try:
            return self._child.wait()
        finally:
            self._child = None

    def stop(self, *_: object) -> None:
        self._stopping = True
        self._arbiter.stop()
        if self._child is not None and self._child.poll() is None:
            # The download or the ingestion; either way the job goes back to the front of the line when the box comes
            # back (recover_interrupted)
            self._child.terminate()

    def run_forever(self) -> None:
        self._arbiter.start()
        recovered = self._queue.recover_interrupted()
        if recovered:
            logger.info(f"Re-queued {recovered} ingestion job(s) interrupted by the last shutdown.")
        self.sweep_uploads()
        while not self._stopping:
            if not self.run_once():
                time.sleep(POLL_SECONDS)


def stale_upload_folders(queue: IngestQueue, tenants_root: Path, now: float,
                         min_age_seconds: float = UPLOAD_SWEEP_AGE_SECONDS) -> List[Path]:
    """Batch folders no queued or running job refers to, untouched for `min_age_seconds`."""
    active = {str(Path(d).resolve()) for d in queue.active_batch_dirs()}
    cutoff = now - min_age_seconds
    return [folder for folder in batch_folders(tenants_root)
            if str(folder.resolve()) not in active and _last_written(folder) <= cutoff]


def batch_folders(tenants_root: Path) -> Iterator[Path]:
    """<tenants>/<firm>/uploads/<user>/<batch>"""
    if not tenants_root.is_dir():
        return
    for firm in sorted(tenants_root.iterdir()):
        uploads = firm / "uploads"
        if not uploads.is_dir():
            continue
        for owner in sorted(uploads.iterdir()):
            if owner.is_dir():
                yield from (batch for batch in sorted(owner.iterdir()) if batch.is_dir() and not batch.is_symlink())


def _last_written(folder: Path) -> float:
    """The newest mtime of the folder and its files: a file still uploading keeps the whole batch fresh."""
    newest = folder.stat().st_mtime
    for path in folder.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            pass
    return newest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from web_deploy.api.config import ApiSettings
    settings = ApiSettings.from_env()
    log_dir = os.environ.get("WEB_LOG_DIR")
    supervisor = Supervisor(IngestQueue.connect(settings.control_database_url), settings.tenants_root,
                            arbiter=arbiter_from_env(dict(os.environ), Path(log_dir) if log_dir else None))
    signal.signal(signal.SIGTERM, supervisor.stop)
    signal.signal(signal.SIGINT, supervisor.stop)
    supervisor.run_forever()


if __name__ == "__main__":
    main()
