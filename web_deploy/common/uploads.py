"""
A firm's upload folders (<tenants>/<firm>/uploads/<user>/<batch>): where a batch is received, what in it can be
ingested, and when it may be removed.

They live on the Hetzner VPS, next to the vault: the gateway receives uploads itself, so a client's file never
lands on the rented GPU box except in the scratch copy of the one job that is ingesting it (ingest_worker/staging.py).
Standard library only, plus the queue passed in: the gateway's venv has no appliance dependencies.
"""

import logging
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterator, List, Protocol, Set

from deploy.instance.agent.uploads import ALLOWED_SUFFIXES, UploadRejected, sanitize_filename
from web_deploy.common.tenancy import firm_root

__all__ = ["ALLOWED_SUFFIXES", "UploadRejected", "sanitize_filename", "BATCH_ID", "new_batch_id", "batch_dir",
           "receive", "batch_files", "UploadJanitor", "batch_folders", "stale_upload_folders"]

logger = logging.getLogger(__name__)

BATCH_ID = re.compile(r"^\d{8}-\d{6}-[a-z0-9]{6}$")
# An upload folder no queued or running job refers to is removed once untouched this long: an upload still arriving
# (its .part file grows) or a batch about to be queued is always newer
UPLOAD_SWEEP_AGE_SECONDS = 24 * 3600


class BatchRejected(ValueError):
    pass


def new_batch_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def batch_dir(tenants_root: Path, firm: str, user_id: str, batch: str) -> Path:
    if not BATCH_ID.match(batch or ""):
        raise BatchRejected(f"'{batch}' is not a batch id.")
    if not re.match(r"^[A-Za-z0-9_-]{1,64}$", user_id or ""):
        raise BatchRejected("Not a user id.")
    return firm_root(tenants_root, firm) / "uploads" / user_id / batch


async def receive(chunks: AsyncIterator[bytes], folder: Path, filename: str, max_bytes: int) -> dict:
    """Streams one uploaded file into the batch folder: a hidden .part file, renamed once it is complete."""
    name = sanitize_filename(filename)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    partial = folder / f".{name}.part"
    size = 0
    try:
        with partial.open("wb") as handle:
            async for chunk in chunks:
                size += len(chunk)
                if size > max_bytes:
                    raise BatchRejected("The file is larger than the appliance accepts.")
                handle.write(chunk)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return {"name": name, "bytes": size}


def batch_files(folder: Path) -> List[Path]:
    """The files of a batch that ingestion will take."""
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES and not p.name.startswith(".")]


class _Queue(Protocol):
    def active_batch_dirs(self) -> Set[str]: ...
    def finished_batch_dirs(self) -> Set[str]: ...


class UploadJanitor:
    """
    Removes upload folders no job will read again. Ingestion copies every file into the firm's vault, so a folder
    would otherwise keep a second copy of everything forever.

    - A finished job's folder (done, failed or cancelled) goes at the next pass.
    - A folder no queued or running job refers to (an upload never queued) goes once untouched for a day.
    A job re-queued by a box stop is queued again, so its folder stays: the retry reads it.
    """

    def __init__(self, queue: _Queue, tenants_root: Path, min_age_seconds: float = UPLOAD_SWEEP_AGE_SECONDS) -> None:
        self._queue = queue
        self._tenants_root = tenants_root
        self._min_age = min_age_seconds

    def run_once(self, now: float) -> int:
        try:
            finished = {str(Path(d).resolve()) for d in self._queue.finished_batch_dirs()}
            active = {str(Path(d).resolve()) for d in self._queue.active_batch_dirs()}
        except Exception as exc:
            logger.warning(f"Upload cleanup skipped: could not read the queue ({exc})")
            return 0
        removed = 0
        for folder in batch_folders(self._tenants_root):
            key = str(folder.resolve())
            if key in active:
                continue
            if key in finished or _last_written(folder) <= now - self._min_age:
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        if removed:
            logger.info(f"Removed {removed} upload folder(s) no job will read again")
        return removed


def stale_upload_folders(active: Set[str], tenants_root: Path, now: float,
                         min_age_seconds: float = UPLOAD_SWEEP_AGE_SECONDS) -> List[Path]:
    """Batch folders not in `active`, untouched for `min_age_seconds`."""
    active = {str(Path(d).resolve()) for d in active}
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

