"""
The uploader's work: which files go, under which names, shrunk where it pays, sent in parallel pieces, resumed.

A video is shrunk to its 720p copy here, on the firm's computer, so only the copy crosses the internet (raw footage
runs to ~100 GB an hour; the copy to about 1). Copies are made one at a time while the previous one uploads, and at
most `ahead` of them wait on disk. A state file remembers the batch and every finished file, so running the same
command again after an interruption continues it: finished files are skipped, a half-sent one resumes from the pieces
the gateway already has.
"""

import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from app.features.storage.utils.upload_record import UploadRecord
from deploy.instance.agent.uploads import (
    AUDIO_SUFFIXES, IMAGE_SUFFIXES, TEXT_SUFFIXES, VIDEO_SUFFIXES, sanitize_filename,
)
from web_deploy.common.video_copy import CopyFailed, CopySettings, copy_name, make_copy
from web_deploy.uploader.client import Gateway

CHUNK_BYTES = 32 * 1024 * 1024
SUFFIXES = {"video": VIDEO_SUFFIXES, "audio": AUDIO_SUFFIXES, "document": TEXT_SUFFIXES, "image": IMAGE_SUFFIXES}

Log = Callable[[str], None]


@dataclass
class Source:
    path: Path
    name: str  # what it is stored and cited as


@dataclass
class Plan:
    mode: str
    language: str
    sources: List[Source]
    shrink: bool = True
    parallel: int = 4
    chunk: int = CHUNK_BYTES
    ahead: int = 2
    copy: CopySettings = field(default_factory=CopySettings)


# ---------- which files, under which names (as the web app names a folder's files) ----------

def collect(paths: List[Path], mode: str) -> Tuple[List[Source], int]:
    """Files of the batch's kind, a folder's subfolders folded into the name; and how many were left out."""
    found: List[Tuple[Path, str]] = []
    skipped = 0
    for root in paths:
        if root.is_dir():
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                relative = path.relative_to(root).parts
                found.append((path, " - ".join(relative)))
        elif root.is_file():
            found.append((root, root.name))
    sources = []
    for path, name in found:
        if path.name.startswith(".") or path.suffix.lower() not in SUFFIXES[mode]:
            skipped += 1
            continue
        sources.append(Source(path, name))
    return _dedupe(sources), skipped


def _dedupe(sources: List[Source]) -> List[Source]:
    """The same name twice becomes "name-2.ext" (sanitised as the gateway will store it)."""
    seen: Dict[str, int] = {}
    for source in sources:
        name = sanitize_filename(source.name)
        stem, suffix = os.path.splitext(name)
        count = seen.get(name.lower(), 0) + 1
        seen[name.lower()] = count
        source.name = name if count == 1 else f"{stem}-{count}{suffix}"
    return sources


# ---------- remembering progress, to resume ----------

class State:
    """One file per batch of inputs: the batch id, and each file already sent (with what it was, if shrunk)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}
        self._lock = threading.Lock()

    @classmethod
    def for_plan(cls, folder: Path, server: str, plan: Plan) -> "State":
        key = json.dumps([server, plan.mode, plan.language, plan.shrink,
                          [(str(s.path.resolve()), s.path.stat().st_size, s.path.stat().st_mtime) for s in plan.sources]])
        folder.mkdir(parents=True, exist_ok=True)
        return cls(folder / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.json")

    @property
    def batch(self) -> Optional[str]:
        return self.data.get("batch")

    def set_batch(self, batch: str) -> None:
        with self._lock:
            self.data = {"batch": batch, "sent": {}, "records": {}}
            self._save()

    def sent(self, source: Source) -> Optional[str]:
        return self.data.get("sent", {}).get(str(source.path))

    def mark_sent(self, source: Source, name: str, record: Optional[UploadRecord]) -> None:
        with self._lock:
            self.data.setdefault("sent", {})[str(source.path)] = name
            if record is not None:
                self.data.setdefault("records", {})[name] = {"sha256": record.sha256,
                                                             "original_name": record.original_name}
            self._save()

    @property
    def records(self) -> dict:
        return dict(self.data.get("records", {}))

    def finish(self) -> None:
        self.path.unlink(missing_ok=True)

    def _save(self) -> None:
        partial = self.path.with_suffix(".tmp")
        partial.write_text(json.dumps(self.data, indent=1))
        partial.replace(self.path)


# ---------- sending ----------

def send_file(gateway: Gateway, batch: str, path: Path, name: str, chunk: int, parallel: int,
              on_bytes: Callable[[int], None]) -> None:
    """One file in pieces, `parallel` at a time, skipping those the gateway already has."""
    size = path.stat().st_size
    status = gateway.status(batch, name, size, chunk)
    count = max(1, -(-size // chunk))
    have = set(range(count)) if status.get("complete") else set(status.get("received", []))
    on_bytes(sum(min(chunk, size - i * chunk) for i in have))
    missing = [i for i in range(count) if i not in have]

    def piece(index: int) -> None:
        with path.open("rb") as handle:
            handle.seek(index * chunk)
            data = handle.read(min(chunk, size - index * chunk))
        gateway.put_chunk(batch, name, index, size, chunk, data)
        on_bytes(len(data))

    if missing:
        with ThreadPoolExecutor(max_workers=min(parallel, len(missing))) as pool:
            for future in [pool.submit(piece, i) for i in missing]:
                future.result()


def run(gateway: Gateway, plan: Plan, state: State, scratch: Path, log: Log = print,
        make: Callable = make_copy) -> dict:
    """Shrinks, sends and queues the batch. Returns the queued job as the gateway describes it."""
    if not state.batch:
        state.set_batch(gateway.new_batch())
        log(f"New batch {state.batch}")
    else:
        log(f"Continuing batch {state.batch}")
    batch = state.batch
    scratch.mkdir(parents=True, exist_ok=True)
    progress = _Progress(log)
    # Names already in use in the batch: the inputs', and those of copies sent before an interruption
    taken = {s.name for s in plan.sources} | {state.sent(s) for s in plan.sources if state.sent(s)}

    # Copies are made one at a time ahead of the upload, never more than `ahead` waiting on disk
    ready: "queue.Queue[Optional[Tuple[Source, Path, str, Optional[UploadRecord], bool]]]" = queue.Queue(plan.ahead)
    failure: List[BaseException] = []

    def produce() -> None:
        try:
            for source in plan.sources:
                if failure:
                    return
                if state.sent(source):
                    continue
                ready.put(_prepare(source, plan, scratch, taken, log, make))
        except BaseException as exc:  # surfaced by the consumer
            failure.append(exc)
        finally:
            ready.put(None)

    threading.Thread(target=produce, daemon=True).start()
    for source in plan.sources:
        if state.sent(source):
            log(f"Already sent: {source.name}")
    while True:
        item = ready.get()
        if item is None:
            break
        source, path, name, record, temporary = item
        try:
            progress.start(name, path.stat().st_size)
            send_file(gateway, batch, path, name, plan.chunk, plan.parallel, progress.add)
            progress.done()
            state.mark_sent(source, name, record)
        except BaseException:
            failure.append(RuntimeError("stopped"))
            raise
        finally:
            if temporary:
                path.unlink(missing_ok=True)
    if failure:
        raise failure[0]

    if state.records:
        gateway.record(batch, state.records)
    job = gateway.enqueue(batch, plan.mode, plan.language)
    state.finish()
    return job


def _prepare(source: Source, plan: Plan, scratch: Path, taken: set, log: Log, make: Callable):
    """(source, file to send, its name, what it was if shrunk, whether the file is ours to delete)."""
    if plan.shrink and plan.mode == "video":
        name = copy_name(Path(source.name), taken - {source.name})
        temp = scratch / f"{hashlib.sha256(str(source.path).encode()).hexdigest()[:16]}.mp4"
        started = time.monotonic()
        try:
            record = make(source.path, temp, plan.copy)
        except CopyFailed as exc:
            log(f"Sending {source.name} as it is: its 720p copy failed ({exc})")
            return source, source.path, source.name, None, False
        if record is not None:
            taken.add(name)
            before, after = source.path.stat().st_size, temp.stat().st_size
            log(f"Shrank {source.name}: {_size(before)} -> {_size(after)} in {time.monotonic() - started:.0f} s")
            return source, temp, name, UploadRecord(record.sha256, source.name), True
    return source, source.path, source.name, None, False


class _Progress:
    def __init__(self, log: Log, every: float = 5.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._log = log
        self._every = every
        self._clock = clock
        self._lock = threading.Lock()

    def start(self, name: str, size: int) -> None:
        self._name, self._size, self._sent = name, size, 0
        self._started = self._last = self._clock()

    def add(self, count: int) -> None:
        with self._lock:
            self._sent += count
            now = self._clock()
            if now - self._last >= self._every:
                self._last = now
                rate = self._sent / max(1e-6, now - self._started)
                self._log(f"  {self._name}: {_size(self._sent)} of {_size(self._size)} ({rate * 8 / 1e6:.0f} Mbit/s)")

    def done(self) -> None:
        seconds = max(1e-6, self._clock() - self._started)
        self._log(f"Sent {self._name} ({_size(self._size)}, {self._size * 8 / 1e6 / seconds:.0f} Mbit/s)")


def _size(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024:
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} TB"

