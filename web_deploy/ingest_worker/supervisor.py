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

The box keeps no client data at rest (docs/web_deploy.md, "Where data lives"). Each job's files are pulled from the
VPS into scratch before it runs and its vault pushed back afterwards (staging.py); upload folders on the VPS are
removed there, by the gateway's janitor (web_deploy/common/uploads.py).

Run: python -m web_deploy.ingest_worker.supervisor  (supervisord, see web_deploy/provision/supervisord.conf)
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from web_deploy.common import youtube
from web_deploy.common.tenancy import database_name, firm_root
from web_deploy.ingest_worker.gpu_arbiter import GpuArbiter, arbiter_from_env
from web_deploy.ingest_worker.queue import IngestJob, IngestQueue, JobOrigin
from web_deploy.ingest_worker.staging import Stager, StagingError, StagingSettings

logger = logging.getLogger(__name__)

POLL_SECONDS = 5.0


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
    # The firm's own glossary for this language, if the owner set one (gateway/glossaries.py; on the hosted box it
    # arrives with the job's files). The child reads <GLOSSARY_PATH>/<language>/<industry>.json exactly as the appliance
    # reads data/glossaries/; without one, the shared general glossary
    env.pop("GLOSSARY_PATH", None)
    env.pop("APPLIANCE_INDUSTRY", None)
    glossary = firm_glossary(root, language)
    if glossary is not None:
        env["GLOSSARY_PATH"] = str(root / "glossaries")
        env["APPLIANCE_INDUSTRY"] = glossary.stem.upper()
    return env


def firm_glossary(firm_dir: Path, language: str) -> Optional[Path]:
    """<firm>/glossaries/<language>/<industry>.json when there is exactly one, else None."""
    files = [p for p in (firm_dir / "glossaries" / language).glob("*.json") if p.name != "meta.json"]
    return files[0] if len(files) == 1 else None


class Supervisor:
    def __init__(self, queue: IngestQueue, tenants_root: Path,
                 spawn: Optional[Callable[[IngestJob, Dict[str, str]], int]] = None,
                 base_env: Optional[Dict[str, str]] = None, arbiter: Optional[GpuArbiter] = None,
                 fetch: Optional[Callable[[IngestJob, Dict[str, str]], int]] = None,
                 stager: Optional[Stager] = None) -> None:
        self._queue = queue
        self._arbiter = arbiter or GpuArbiter()
        self._tenants_root = tenants_root
        self._spawn = spawn or self._spawn_child
        self._fetch = fetch or self._spawn_fetch
        self._base_env = dict(base_env if base_env is not None else os.environ)
        # Off (one shared disk) unless BOX_GATEWAY_URL names the gateway through the tunnel
        self._stager = stager or Stager(StagingSettings(gateway_url="", tenants_root=tenants_root,
                                                        state_dir=tenants_root / ".staging"))
        self._child: Optional[subprocess.Popen] = None
        self._stopping = False

    def run_once(self) -> bool:
        """Runs the next job if there is one. Returns whether it ran one."""
        job = self._queue.claim_next()
        if job is None:
            self._arbiter.when_idle()
            return False
        try:
            if self._stage_in(job):
                self._run_job(job)
        finally:
            self._stage_out(job)
        return True

    def _stage_in(self, job: IngestJob) -> bool:
        """The batch's files into the box's scratch. Returns whether the job should run."""
        try:
            self._stager.stage_in(job.id, job.firm, None if job.origin is JobOrigin.YOUTUBE else job.batch_dir)
            if job.origin is JobOrigin.YOUTUBE and job.manifest:
                youtube.place_manifest(Path(job.batch_dir), job.manifest)
            return True
        except Exception as exc:
            logger.exception(f"Could not bring job {job.id}'s files to the box")
            self._queue.fail(job.id, f"The files could not be fetched for ingestion: {exc}")
            return False

    def _stage_out(self, job: IngestJob) -> None:
        """
        The vault back to the VPS and the scratch wiped, whatever the job's outcome: a failed job's rows are in the
        database already and must not point at files that only ever existed here. Not during a stop: the marker
        stays, and the next start pushes it (Stager.recover) before the re-queued job runs again.
        """
        if self._stopping:
            return
        try:
            self._stager.stage_out(job.id)
        except StagingError as exc:
            # /health keeps the box awake while the marker is there; the next pass of the loop retries
            logger.error(f"Could not push job {job.id}'s results to the VPS: {exc}")

    def _run_job(self, job: IngestJob) -> None:
        if job.origin is JobOrigin.YOUTUBE:
            try:
                downloaded = self._download(job)
            finally:
                # The refreshed cookies go back to the VPS (or the owner hears YouTube refused them); none stay here
                self._stager.youtube_cookies_back(job.id, refused=youtube.bot_checked(Path(job.batch_dir)))
                self._stager.drop_youtube_cookies()
            if not downloaded:
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
        self._allow_staging()
        self._recover_staging()
        recovered = self._queue.recover_interrupted()
        if recovered:
            logger.info(f"Re-queued {recovered} ingestion job(s) interrupted by the last shutdown.")
        while not self._stopping:
            if self._stager.has_pending() and not self._recover_staging():
                time.sleep(POLL_SECONDS)  # a push still pending: the VPS's files first, new work after
                continue
            if not self.run_once():
                time.sleep(POLL_SECONDS)

    def _allow_staging(self) -> None:
        """Lets the VPS reach the scratch (its staging key, for root). Retried until the gateway answers."""
        while not self._stopping:
            try:
                self._stager.ensure_access()
                return
            except Exception as exc:
                logger.error(f"Could not let the VPS's staging key in yet: {exc}")
                time.sleep(POLL_SECONDS)

    def _recover_staging(self) -> bool:
        """Pushes what an earlier run left in scratch. Returns whether nothing is pending any more."""
        try:
            return self._stager.recover() == 0
        except Exception as exc:
            logger.error(f"Could not recover the box's scratch: {exc}")
            return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from web_deploy.common.box_credentials import install_from_env
    install_from_env()  # the hosted box: the database password comes from the gateway, never from the environment
    from web_deploy.api.config import ApiSettings
    settings = ApiSettings.from_env()
    log_dir = os.environ.get("WEB_LOG_DIR")
    supervisor = Supervisor(IngestQueue.connect(settings.control_database_url), settings.tenants_root,
                            arbiter=arbiter_from_env(dict(os.environ), Path(log_dir) if log_dir else None),
                            stager=Stager(StagingSettings.from_env()))
    signal.signal(signal.SIGTERM, supervisor.stop)
    signal.signal(signal.SIGINT, supervisor.stop)
    supervisor.run_forever()


if __name__ == "__main__":
    main()
