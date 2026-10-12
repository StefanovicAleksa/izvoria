"""
Brings one ingestion job's files to the GPU box and its results back, so the box keeps no client data at rest.

Every firm's vault and uploads live on the Hetzner VPS (docs/web_deploy.md, "Where data lives"). The box only has
a scratch copy at the SAME absolute path (TENANTS_ROOT, /workspace/tenants on both machines), because the database
stores absolute file paths: a path the ingestion child writes on the box is valid on the VPS once pushed.

A job, around the ingestion child (Supervisor._run_job):
1. `stage_in`: the batch's uploads, pulled from the VPS. Then every vault file that belongs to a source with the
   same content as one of them (the original, its frames, a kept WAV or clips), so a retry, a re-upload or a
   half-finished earlier run finds its files exactly where the database says they are.
2. the ingestion child writes the vault in scratch, and its rows straight into the VPS database.
3. `stage_out`: the firm's scratch vault pushed to the VPS, the vault files the job deleted (a compaction swap of a
   pulled original) deleted there too, then the scratch wiped.

A marker per job (STAGING_STATE_DIR) says what was pulled. It lives until the push succeeded, so a crash, a host
stop or a failed push leaves the job recoverable: `recover()` pushes and wipes at the next start, and the API's
/health reports `staging_pending` meanwhile, which keeps the gateway's idle stop away.

All transfers are rsync over SSH to the VPS's `izvoria-box` account, whose key is confined there to
`rrsync /workspace/tenants` (install_gateway.sh): remote paths are relative to the tenants root.

Rows the child writes reach the VPS database before their files reach the VPS vault; until the push a citation to
that new file is "not available yet" on the gateway, never an error.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from app.features.storage.utils.upload_record import read_record
from web_deploy.common.tenancy import firm_root

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


class StagingError(RuntimeError):
    pass


@dataclass(frozen=True)
class StagingSettings:
    """From api.env. Without STAGING_REMOTE the box and the data share one disk (dev hardware): staging is off."""
    remote: str
    tenants_root: Path
    state_dir: Path
    ssh_key: Optional[Path] = None
    ssh_port: int = 22
    known_hosts: Optional[Path] = None

    @property
    def enabled(self) -> bool:
        return bool(self.remote)

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "StagingSettings":
        env = dict(os.environ if env is None else env)
        key = env.get("STAGING_SSH_KEY", "")
        known = env.get("STAGING_KNOWN_HOSTS", "")
        return cls(
            remote=env.get("STAGING_REMOTE", "").strip(),
            tenants_root=Path(env.get("TENANTS_ROOT", "/workspace/tenants")),
            state_dir=Path(env.get("STAGING_STATE_DIR", "/workspace/staging-state")),
            ssh_key=Path(key) if key else None,
            ssh_port=int(env.get("STAGING_SSH_PORT", "22") or 22),
            known_hosts=Path(known) if known else None,
        )


def pending_markers(state_dir: Path) -> List[Path]:
    """Jobs whose results are not on the VPS yet. The API's /health reads this."""
    return sorted(state_dir.glob("*.json")) if state_dir.is_dir() else []


def sha256_of(path: Path) -> str:
    """The content hash the storage feature dedups on (app/features/storage/utils/hashing.py: sha256, hex)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class Stager:
    def __init__(self, settings: StagingSettings, lookup: Optional[VaultLookup] = None,
                 run: Runner = subprocess.run) -> None:
        self._settings = settings
        self._lookup = lookup or self._lookup_in_database
        self._run = run

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def has_pending(self) -> bool:
        return self.enabled and bool(pending_markers(self._settings.state_dir))

    # ---------- a job ----------

    def stage_in(self, job_id: str, firm: str, batch_dir: Optional[str]) -> List[str]:
        """
        Pulls the batch (None for a YouTube batch, which is downloaded on the box) and the vault files of matching
        sources. Returns the vault paths pulled, which the marker keeps.
        """
        if not self.enabled:
            return []
        self._write_marker(job_id, {"firm": firm, "batch_dir": batch_dir, "pulled": []})
        if batch_dir:
            local = self._inside(Path(batch_dir))
            local.mkdir(parents=True, exist_ok=True)
            self._rsync([f"{self._remote(local)}/", f"{local}/"])
            # A 720p copy the VPS made is known by its upload's hash (upload_record.py), as ingestion will know it
            hashes = []
            for path in sorted(local.iterdir()):
                if path.is_file() and not path.name.startswith("."):
                    record = read_record(path)
                    hashes.append(record.sha256 if record else sha256_of(path))
        else:
            hashes = []
        pulled = self._pull_vault_files(firm, hashes)
        self._write_marker(job_id, {"firm": firm, "batch_dir": batch_dir, "pulled": pulled})
        return pulled

    def stage_out(self, job_id: str) -> None:
        """
        Pushes the firm's scratch vault, deletes on the VPS what the job deleted, then wipes the scratch. Raises
        StagingError and keeps everything when the push fails: the marker makes `recover` try again.
        """
        if not self.enabled:
            return
        marker = self._read_marker(job_id)
        if marker is None:
            return
        firm = marker["firm"]
        vault = firm_root(self._settings.tenants_root, firm) / "vault"
        if vault.is_dir():
            self._rsync([f"{vault}/", f"{self._remote(vault)}/"])
        deleted = [p for p in marker.get("pulled", []) if not Path(p).exists()]
        if deleted:
            self._delete_remote(deleted)
        self.wipe(firm)
        (self._settings.state_dir / f"{job_id}.json").unlink(missing_ok=True)
        logger.info(f"Job {job_id}: results pushed to the VPS and the box's copy wiped"
                    + (f" ({len(deleted)} replaced file(s) removed there)" if deleted else ""))

    def recover(self) -> int:
        """
        At start: pushes what a crash, a host stop or a failed push left behind, then wipes every firm's scratch.
        Returns how many jobs were still pending (still pending after this when the VPS is unreachable).
        """
        if not self.enabled:
            return 0
        markers = pending_markers(self._settings.state_dir)
        for marker in markers:
            try:
                self.stage_out(marker.stem)
            except StagingError as exc:
                logger.error(f"Could not push job {marker.stem}'s results to the VPS yet: {exc}")
        left = pending_markers(self._settings.state_dir)
        if not left:
            self.wipe_all()
        return len(left)

    # ---------- scratch ----------

    def wipe(self, firm: str) -> None:
        """A firm's scratch vault and uploads (reports are kept in the database only: REPORT_FILES_ENABLED=false)."""
        root = firm_root(self._settings.tenants_root, firm)
        for part in ("vault", "uploads"):
            _remove_tree(root / part)
            (root / part).mkdir(parents=True, exist_ok=True)

    def wipe_all(self) -> None:
        root = self._settings.tenants_root
        if not root.is_dir():
            return
        for firm in sorted(root.iterdir()):
            if firm.is_dir() and not firm.is_symlink():
                for part in ("vault", "uploads", "reports"):
                    _remove_tree(firm / part)
                    (firm / part).mkdir(parents=True, exist_ok=True)

    # ---------- plumbing ----------

    def _pull_vault_files(self, firm: str, hashes: List[str]) -> List[str]:
        if not hashes:
            return []
        paths = sorted({str(self._inside(Path(p))) for p in self._lookup(firm, hashes)})
        if not paths:
            return []
        root = self._settings.tenants_root
        with tempfile.NamedTemporaryFile("w", suffix=".list", delete=False) as listing:
            listing.write("\n".join(str(Path(p).relative_to(root)) for p in paths) + "\n")
        try:
            # Files the database names but the VPS lacks (a push that never happened) are skipped, not fatal: the
            # pipeline re-creates what it can, exactly as it would after a manual deletion
            self._rsync(["--ignore-missing-args", f"--files-from={listing.name}",
                         f"{self._settings.remote}:./", f"{root}/"], check_partial=True)
        finally:
            os.unlink(listing.name)
        arrived = [p for p in paths if Path(p).is_file()]
        logger.info(f"Pulled {len(arrived)} vault file(s) of already-known sources for {firm}")
        return arrived

    def _delete_remote(self, paths: Sequence[str]) -> None:
        """
        Deletes these files on the VPS: a `--delete` push filtered down to exactly them. Every other file is excluded,
        and rsync never deletes an excluded file, so nothing else there can go. (rrsync refuses
        --delete-missing-args, the direct way.)
        """
        root = self._settings.tenants_root
        rules: List[str] = []
        for path in paths:
            relative = Path(path).relative_to(root)
            # The sender must have each parent folder for rsync to descend into it on both sides
            (root / relative.parent).mkdir(parents=True, exist_ok=True)
            for depth in range(1, len(relative.parts)):
                rule = "+ /" + "/".join(relative.parts[:depth]) + "/"
                if rule not in rules:
                    rules.append(rule)
            rules.append("+ /" + relative.as_posix())
        rules.append("- *")
        with tempfile.NamedTemporaryFile("w", suffix=".rules", delete=False) as listing:
            listing.write("\n".join(rules) + "\n")
        try:
            self._rsync(["--delete", f"--filter=merge {listing.name}", f"{root}/", f"{self._settings.remote}:./"])
        finally:
            os.unlink(listing.name)

    def _rsync(self, arguments: List[str], check_partial: bool = False) -> None:
        command = ["rsync", "-rlt", "--chmod=D2770,F640", "-e", self._ssh_command(), *arguments]
        completed = self._run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        # 23/24: some files vanished or were missing; acceptable only where missing files are expected
        if completed.returncode != 0 and not (check_partial and completed.returncode in (23, 24)):
            raise StagingError(f"rsync failed ({completed.returncode}): {(completed.stderr or '').strip()[-800:]}")

    def _ssh_command(self) -> str:
        parts = ["ssh", "-p", str(self._settings.ssh_port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                 "-o", "ServerAliveInterval=15"]
        if self._settings.ssh_key is not None:
            parts += ["-i", str(self._settings.ssh_key), "-o", "IdentitiesOnly=yes"]
        if self._settings.known_hosts is not None:
            parts += ["-o", f"UserKnownHostsFile={self._settings.known_hosts}"]
        return " ".join(parts)

    def _remote(self, local: Path) -> str:
        """The VPS side of a scratch path: relative to the tenants root, which rrsync confines the key to."""
        return f"{self._settings.remote}:{local.relative_to(self._settings.tenants_root)}"

    def _inside(self, path: Path) -> Path:
        """`path` if it lies under the tenants root, else refuses: nothing outside it is ever sent or deleted."""
        root = self._settings.tenants_root
        resolved = Path(os.path.normpath(path))
        if resolved == root or root not in resolved.parents:
            raise StagingError(f"{path} is not under {root}")
        return resolved

    def _write_marker(self, job_id: str, marker: dict) -> None:
        self._settings.state_dir.mkdir(parents=True, exist_ok=True)
        target = self._settings.state_dir / f"{job_id}.json"
        partial = target.with_suffix(".tmp")
        partial.write_text(json.dumps(marker))
        partial.replace(target)

    def _read_marker(self, job_id: str) -> Optional[dict]:
        try:
            return json.loads((self._settings.state_dir / f"{job_id}.json").read_text())
        except (OSError, ValueError):
            return None

    def _lookup_in_database(self, firm: str, hashes: List[str]) -> List[str]:
        import psycopg2
        from web_deploy.common.tenancy import database_name
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


def _remove_tree(path: Path) -> None:
    """Vault files are stored read-only (0o644 in a 0o755 tree); rmtree as their owner removes them regardless."""
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)


def staging_from_env() -> Stager:
    return Stager(StagingSettings.from_env())

