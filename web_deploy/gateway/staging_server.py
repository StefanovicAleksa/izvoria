"""
The VPS's side of an ingestion job's staging: it sends one job's files to the GPU box and takes the results back.

The box has no key that can reach the VPS's files. The VPS reaches the box instead, through a port the box forwards
here (its sshd at 127.0.0.1:BOX_FILES_PORT, on a tunnel connection of its own, so a large transfer never delays the
database traffic chat runs on). The box asks for the transfers (gateway/box_internal.py), but the VPS decides what
moves: only for the job that is running, only that job's batch and the vault files of sources with the same content,
and back only that firm's scratch vault. Paths are the same on both machines (/workspace/tenants): the database stores
absolute file paths.

1. `stage_in(job)`: the batch (unless it is a YouTube batch, downloaded on the box), then every vault file of a source
   whose content matches an upload (its original, frames, a kept WAV, clips), so a retry, a re-upload or a
   half-finished earlier run finds its files where the database says they are. What was sent is remembered.
2. `stage_out(job)`: the firm's scratch vault, back; then the files sent in step 1 that the box no longer has (a
   compaction swap of a pulled original) are deleted here too.
"""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.features.storage.utils.upload_record import read_record
from web_deploy.common.tenancy import database_name, firm_root
from web_deploy.common.uploads import batch_files
from web_deploy.gateway import youtube_cookies
from web_deploy.ingest_worker.queue import IngestJob, IngestQueue, JobOrigin, JobStatus

logger = logging.getLogger(__name__)

HASH_CHUNK_BYTES = 8 * 1024 * 1024

# Every vault file of the sources whose content matches an upload: the original, its extracted audio, per-chunk clips
# and the frames visual analysis reads (file ids inside extracted_chunk_frames.frames)
VAULT_FILES_SQL = """
WITH matched AS (SELECT id, file_id FROM sources WHERE content_hash = ANY(%(hashes)s))
SELECT f.file_path FROM files f WHERE f.id IN (
    SELECT file_id FROM matched
    UNION SELECT extracted_file_id FROM audio_extractions WHERE source_id IN (SELECT id FROM matched)
    UNION SELECT extracted_file_id FROM video_clips WHERE source_id IN (SELECT id FROM matched)
    UNION SELECT (frame ->> 'file_id')::uuid
          FROM extracted_chunk_frames e, jsonb_array_elements(
              CASE WHEN jsonb_typeof(e.frames) = 'array' THEN e.frames ELSE '[]'::jsonb END) AS frame
          WHERE e.source_id IN (SELECT id FROM matched) AND frame ? 'file_id'
)
"""

Runner = Callable[..., subprocess.CompletedProcess]
# (firm, content hashes) -> absolute vault paths the firm's database knows for them
VaultLookup = Callable[[str, List[str]], List[str]]


class StagingRefused(ValueError):
    """Not a job this box may move files for (unknown, another state, or a path outside the tenants)."""


class StagingFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class BoxFiles:
    """How the VPS reaches the box's files: its sshd, forwarded to this port, with the VPS's own staging key."""
    tenants_root: Path
    state_dir: Path
    port: int = 18022
    key: Optional[Path] = None
    # The owner's YouTube cookies (youtube_cookies.py), sent for a YouTube job only, and where the box reads them
    youtube_cookies: Optional[Path] = None
    box_youtube_cookies: Path = Path("/workspace/web/youtube_cookies.txt")

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "BoxFiles":
        env = dict(os.environ if env is None else env)
        key = env.get("BOX_STAGING_KEY", "/etc/onyx-web/staging_ed25519")
        return cls(
            tenants_root=Path(env.get("TENANTS_ROOT", "/workspace/tenants")),
            state_dir=Path(env.get("BOX_STAGING_STATE_DIR", "/var/lib/onyx-web/staging")),
            port=int(env.get("BOX_FILES_PORT", "18022")),
            key=Path(key) if key else None,
        )


def sha256_of(path: Path) -> str:
    """The hash ingestion dedups a source on (app/features/storage/utils/hashing.py: sha256, hex)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class StagingServer:
    def __init__(self, files: BoxFiles, queue: IngestQueue, lookup: Optional[VaultLookup] = None,
                 run: Runner = subprocess.run, on_youtube_refused: Optional[Callable[[], None]] = None) -> None:
        self._files = files
        self._queue = queue
        self._lookup = lookup or self._lookup_in_database
        self._run = run
        self._on_youtube_refused = on_youtube_refused

    # ---------- the two transfers ----------

    def stage_in(self, job_id: str) -> dict:
        job = self._job(job_id, allowed=(JobStatus.RUNNING,))
        sent_batch = 0
        hashes: List[str] = []
        if job.origin is not JobOrigin.YOUTUBE:
            batch = self._inside(Path(job.batch_dir))
            if not batch.is_dir():
                raise StagingFailed(f"The batch folder of job {job.id} is gone from the VPS.")
            self._rsync(["--mkpath", f"{batch}/", self._remote(batch) + "/"])
            files = batch_files(batch)
            sent_batch = len(files)
            for path in files:
                record = read_record(path)
                hashes.append(record.sha256 if record else sha256_of(path))
        vault = self._send_vault_files(job.firm, hashes)
        # The firm's own glossary for the job's language, if the owner set one (gateway/glossaries.py)
        glossary = firm_root(self._files.tenants_root, job.firm) / "glossaries" / job.language
        if any(glossary.glob("*.json")):
            self._rsync(["--mkpath", f"{glossary}/", self._remote(glossary) + "/"])
        # YouTube refuses the box's data-centre IP without a signed-in account; the box deletes this after the download
        cookies = self._files.youtube_cookies
        if job.origin is JobOrigin.YOUTUBE and cookies is not None and cookies.is_file():
            self._call(["rsync", "-t", "--chmod=F600", "-e", self._ssh(), str(cookies),
                        self._remote(self._files.box_youtube_cookies)])
        self._remember(job_id, {"firm": job.firm, "sent": vault})
        logger.info(f"Job {job_id}: sent {sent_batch} batch file(s) and {len(vault)} vault file(s) to the box")
        return {"batch_files": sent_batch, "vault_files": len(vault)}

    def stage_out(self, job_id: str) -> dict:
        job = self._job(job_id, allowed=(JobStatus.RUNNING, JobStatus.DONE, JobStatus.FAILED, JobStatus.QUEUED))
        vault = firm_root(self._files.tenants_root, job.firm) / "vault"
        vault.mkdir(parents=True, exist_ok=True)
        # Only the firm's vault comes back; a box can't write anywhere else here
        self._rsync([self._remote(vault) + "/", f"{vault}/"])
        on_box = self._list_remote(vault)
        sent = self._recall(job_id).get("sent", [])
        removed = 0
        for path in sent:
            relative = str(Path(path).relative_to(vault))
            if relative not in on_box:
                Path(path).unlink(missing_ok=True)
                removed += 1
        self._forget(job_id)
        logger.info(f"Job {job_id}: results taken back from the box"
                    + (f"; {removed} replaced file(s) removed here" if removed else ""))
        return {"removed": removed}

    # ---------- plumbing ----------

    # ---------- the YouTube account, after a download step ----------

    def youtube_cookies_refreshed(self, job_id: str) -> dict:
        """
        The cookie jar yt-dlp saved on the box, taken back when it is still a signed-in session: YouTube rotates
        session cookies as they are used, so this is what keeps the sign-in from going stale.
        """
        job = self._job(job_id, allowed=(JobStatus.RUNNING, JobStatus.DONE, JobStatus.FAILED, JobStatus.QUEUED))
        stored = self._files.youtube_cookies
        if job.origin is not JobOrigin.YOUTUBE or stored is None or not stored.is_file():
            return {"taken_back": False}
        self._files.state_dir.mkdir(parents=True, exist_ok=True)
        handle, partial = tempfile.mkstemp(prefix=".youtube_cookies-", dir=self._files.state_dir)
        os.close(handle)
        try:
            try:
                self._call(["rsync", "-t", "-e", self._ssh(), self._remote(self._files.box_youtube_cookies), partial])
            except StagingFailed as exc:
                logger.warning(f"Job {job_id}: no YouTube cookies to take back from the box: {exc}")
                return {"taken_back": False}
            text = Path(partial).read_text(encoding="utf-8", errors="replace")
            taken = youtube_cookies.take_back(stored, text)
        finally:
            Path(partial).unlink(missing_ok=True)
        logger.info(f"Job {job_id}: YouTube cookies {'refreshed' if taken else 'kept as they were'}")
        return {"taken_back": taken}

    def youtube_refused(self, job_id: str) -> dict:
        """YouTube took the box for a bot: the cookies are missing or stale. The owner is emailed, once a day at most."""
        job = self._job(job_id, allowed=(JobStatus.RUNNING, JobStatus.DONE, JobStatus.FAILED, JobStatus.QUEUED))
        stored = self._files.youtube_cookies
        if job.origin is not JobOrigin.YOUTUBE or stored is None:
            return {"alerted": False}
        news = youtube_cookies.mark_refused(stored)
        logger.warning(f"Job {job_id}: YouTube refused the box's downloads as a bot")
        if news and self._on_youtube_refused is not None:
            try:
                self._on_youtube_refused()
            except Exception:  # noqa: BLE001 - the alert is a courtesy; the job's own error already says it
                logger.exception("Could not alert the owner about the YouTube cookies")
                return {"alerted": False}
        return {"alerted": news}

    def _job(self, job_id: str, allowed) -> IngestJob:
        job = self._queue.get(job_id)
        if job is None or job.status not in allowed:
            raise StagingRefused(f"Job {job_id} is not one the box may move files for now.")
        return job

    def _send_vault_files(self, firm: str, hashes: List[str]) -> List[str]:
        if not hashes:
            return []
        root = self._files.tenants_root
        vault = firm_root(root, firm) / "vault"
        paths = sorted({str(p) for p in map(Path, self._lookup(firm, hashes))
                        if p.is_file() and vault in p.parents})
        if not paths:
            return []
        with tempfile.NamedTemporaryFile("w", suffix=".list", delete=False) as listing:
            listing.write("\n".join(str(Path(p).relative_to(root)) for p in paths) + "\n")
        try:
            self._rsync([f"--files-from={listing.name}", f"{root}/", self._remote(root) + "/"])
        finally:
            os.unlink(listing.name)
        return paths

    def _list_remote(self, vault: Path) -> set:
        """Relative paths of every file in the box's copy of this vault."""
        completed = self._call(["rsync", "-r", "--list-only", "-e", self._ssh(), self._remote(vault) + "/"])
        found = set()
        for line in (completed.stdout or "").splitlines():
            # "-rw-r--r--   1,234 2026/10/12 12:00:00 2026/10/12/<uuid>.mp4"
            parts = line.split(None, 4)
            if len(parts) == 5 and parts[0].startswith("-"):
                found.add(parts[4])
        return found

    def _rsync(self, arguments: List[str]) -> None:
        self._call(["rsync", "-rlt", "--chmod=D2770,F640", "-e", self._ssh(), *arguments])

    def _call(self, command: List[str]) -> subprocess.CompletedProcess:
        completed = self._run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        if completed.returncode != 0:
            raise StagingFailed(f"rsync failed ({completed.returncode}): {(completed.stderr or '').strip()[-800:]}")
        return completed

    def _ssh(self) -> str:
        # The connection runs inside the box's own authenticated tunnel, to a port only it can listen on; a new box
        # has a new host key, so it is not pinned here
        parts = ["ssh", "-p", str(self._files.port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
        if self._files.key is not None:
            parts += ["-i", str(self._files.key), "-o", "IdentitiesOnly=yes"]
        return " ".join(parts)

    def _remote(self, local: Path) -> str:
        return f"root@127.0.0.1:{local}"

    def _inside(self, path: Path) -> Path:
        root = self._files.tenants_root
        resolved = Path(os.path.normpath(path))
        if root not in resolved.parents:
            raise StagingRefused(f"{path} is not under {root}")
        return resolved

    def _remember(self, job_id: str, state: dict) -> None:
        self._files.state_dir.mkdir(parents=True, exist_ok=True)
        target = self._files.state_dir / f"{job_id}.json"
        partial = target.with_suffix(".tmp")
        partial.write_text(json.dumps(state))
        partial.replace(target)

    def _recall(self, job_id: str) -> dict:
        try:
            return json.loads((self._files.state_dir / f"{job_id}.json").read_text())
        except (OSError, ValueError):
            return {}

    def _forget(self, job_id: str) -> None:
        (self._files.state_dir / f"{job_id}.json").unlink(missing_ok=True)

    def _lookup_in_database(self, firm: str, hashes: List[str]) -> List[str]:
        import psycopg2
        with psycopg2.connect(host=os.environ.get("POSTGRES_SERVER", "127.0.0.1"),
                              port=os.environ.get("POSTGRES_PORT", "5432"),
                              user=os.environ.get("POSTGRES_USER", "onyx"),
                              password=os.environ.get("POSTGRES_PASSWORD", ""),
                              dbname=database_name(firm), connect_timeout=15) as connection:
            with connection.cursor() as cursor:
                try:
                    cursor.execute(VAULT_FILES_SQL, {"hashes": hashes})
                except psycopg2.errors.UndefinedTable:
                    return []  # a firm whose database has never been migrated has no sources yet
                return [row[0] for row in cursor.fetchall()]
