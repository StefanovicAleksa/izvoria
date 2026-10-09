"""
Backs the GPU box up to a Hetzner Storage Box: every firm's database (pg_dump), every firm's vault and reports, and
the box's own settings and TLS key. Runs under supervisord beside the API and the ingestion worker.

A backup runs once BACKUP_INTERVAL_HOURS have passed since the last good one, and as soon as an ingestion job has
finished since then, so new material is off the box before the gateway stops it. While one is due or running the
API's /health reports busy, which keeps the gateway's idle stop away. A run that failed is retried after an hour,
not at once, so a broken Storage Box cannot keep a rented GPU awake for good.

Uploads are not backed up (their files are in the vault), and neither is the source cache.

Run: python -m web_deploy.backup.runner          (supervisord)
     python -m web_deploy.backup.runner --now    (one backup, now)
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from web_deploy.backup.config import BackupSettings
from web_deploy.backup.restic import Restic

logger = logging.getLogger(__name__)

TAG = "box"
RETRY_AFTER_SECONDS = 3600
POLL_SECONDS = 60
CONTROL_DATABASE = "onyx_control"
FIRM_DATABASE_PATTERN = "onyx\\_firm\\_%"


# ---------- status: shared with the API's /health through one small file ----------

def read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, path)


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def is_running(status: dict) -> bool:
    return bool(status.get("running")) and _alive(status.get("pid"))


def is_due(settings: BackupSettings, status: dict, last_finished: Optional[datetime], now: float) -> bool:
    """Whether a backup should run now (the queue being idle is checked separately)."""
    if not settings.configured:
        return False
    last_attempt = status.get("last_attempt")
    if status.get("last_error") and last_attempt and now - last_attempt < RETRY_AFTER_SECONDS:
        return False
    last_success = status.get("last_success")
    if last_success is None:
        return True
    if now - last_success >= settings.interval_hours * 3600:
        return True
    return last_finished is not None and last_finished.timestamp() > last_success


def health_state(settings: BackupSettings, last_finished: Optional[datetime], now: float) -> dict:
    """What /health adds: a backup running, or due (about to start), keeps the box awake."""
    status = read_status(settings.status_file)
    running = is_running(status)
    return {"backup_running": running, "backup_due": not running and is_due(settings, status, last_finished, now)}


# ---------- the backup ----------

class BackupRunner:
    def __init__(self, settings: BackupSettings, queue, restic: Optional[Restic] = None,
                 run: Callable = subprocess.run, clock: Callable[[], float] = time.time) -> None:
        self._settings = settings
        self._queue = queue
        self._restic = restic or Restic(settings, run=run)
        self._run = run
        self._clock = clock
        self._stopping = False

    def due(self) -> bool:
        depth = self._queue.depth()
        if depth["queued"] or depth["running"]:
            return False  # after the queue empties: one consistent snapshot of the vault and its databases
        status = read_status(self._settings.status_file)
        return is_due(self._settings, status, self._queue.last_finished_at(), self._clock())

    def run_once(self) -> bool:
        """One backup. Returns whether it succeeded; the outcome is in the status file either way."""
        status = read_status(self._settings.status_file)
        started = self._clock()
        status.update({"running": True, "pid": os.getpid(), "last_attempt": started})
        write_status(self._settings.status_file, status)
        try:
            self._restic.unlock()
            self._restic.ensure_repository()
            databases = self.dump_databases()
            self._restic.backup(self.paths(), self.excludes(), TAG)
            self._restic.forget(TAG)
            status.update({"last_success": self._clock(), "last_error": None, "databases": databases,
                           "last_duration_seconds": round(self._clock() - started, 1)})
            logger.info(f"Backup done: {len(databases)} database(s) and the vaults, "
                        f"in {status['last_duration_seconds']} s")
            return True
        except Exception as exc:
            status["last_error"] = str(exc)[-2000:]
            logger.error(f"Backup failed: {exc}")
            return False
        finally:
            # The dumps are in the snapshot now (or the run failed): no need to keep a second copy on the box
            shutil.rmtree(self._settings.staging / "databases", ignore_errors=True)
            status["running"] = False
            write_status(self._settings.status_file, status)

    def paths(self) -> List[Path]:
        paths = [self._settings.tenants_root, self._settings.staging]
        return paths + [p for p in self._settings.extra_paths if p.exists()]

    def excludes(self) -> List[str]:
        # Uploads are copies of what the vault holds; the status file changes during the run itself
        return [str(self._settings.tenants_root / "*" / "uploads"), str(self._settings.status_file),
                str(self._settings.status_file.with_suffix(".tmp"))]

    def dump_databases(self) -> List[str]:
        """pg_dump of the control database and every firm's, into the staging folder restic then reads."""
        target = self._settings.staging / "databases"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        names = self._databases()
        for name in names:
            self._pg(["pg_dump", "--format=custom", "--no-owner", "--file", str(target / f"{name}.dump"), name])
        return names

    def _databases(self) -> List[str]:
        out = self._pg(["psql", "-d", "postgres", "-At", "-c",
                        f"SELECT datname FROM pg_database WHERE datname = '{CONTROL_DATABASE}' "
                        f"OR datname LIKE '{FIRM_DATABASE_PATTERN}' ORDER BY datname"]).stdout
        return [line.strip() for line in (out or "").splitlines() if line.strip()]

    def _pg(self, command: List[str]) -> subprocess.CompletedProcess:
        pg = self._settings.postgres
        env = dict(os.environ, PGPASSWORD=pg["password"])
        completed = self._run([command[0], "-h", pg["host"], "-p", pg["port"], "-U", pg["user"], *command[1:]],
                              env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"{command[0]} failed: {(completed.stderr or '').strip()[-1000:]}")
        return completed

    def stop(self, *_: object) -> None:
        self._stopping = True

    def run_forever(self) -> None:
        status = read_status(self._settings.status_file)
        if status.get("running") and not _alive(status.get("pid")):
            # The box stopped mid-run: the next run starts over (restic keeps what it had already sent)
            status["running"] = False
            write_status(self._settings.status_file, status)
        if not self._settings.configured:
            logger.info("Backups are off (BACKUP_ENABLED, RESTIC_REPOSITORY, RESTIC_PASSWORD); idling.")
        while not self._stopping:
            try:
                if self.due():
                    self.run_once()
            except Exception as exc:
                logger.error(f"Backup check failed: {exc}")
            for _ in range(POLL_SECONDS):
                if self._stopping:
                    break
                time.sleep(1)


def main(argv: Optional[list] = None) -> int:
    import signal
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from web_deploy.api.config import ApiSettings
    from web_deploy.ingest_worker.queue import IngestQueue
    settings = BackupSettings.from_env()
    queue = IngestQueue.connect(ApiSettings.from_env().control_database_url)
    runner = BackupRunner(settings, queue)
    if "--now" in (argv if argv is not None else sys.argv[1:]):
        if not settings.configured:
            print("error: set BACKUP_ENABLED=true, RESTIC_REPOSITORY and RESTIC_PASSWORD first.", file=sys.stderr)
            return 1
        return 0 if runner.run_once() else 1
    signal.signal(signal.SIGTERM, runner.stop)
    signal.signal(signal.SIGINT, runner.stop)
    runner.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
