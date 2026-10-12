"""
Resumable, parallel uploads: a file arrives as numbered pieces, in any order and several at once, and an upload that
breaks off continues from the pieces already here.

One connection to a server far away rarely fills a fast office line, and a 50 GB recording sent in one request starts
over from zero when the connection drops. So a client (the web app's Ingest page, or the firm's own uploader,
web_deploy/uploader) asks which pieces of a file are here, then sends the missing ones, several at a time:

    POST /api/uploads/{batch}/{name}/chunks      {"size": S, "chunk": C}  -> {"received": [0, 1, 4], "complete": false}
    PUT  /api/uploads/{batch}/{name}/chunks/{i}  bytes of piece i         -> {"complete": false}

Piece i is bytes [i*C, min((i+1)*C, S)). Pieces are written at their offset into one hidden file of the full size,
so finishing costs a rename, never a copy; a marker file per piece says it arrived whole. A different size or piece
size for the same name starts that file over. Standard library only.
"""

import asyncio
import json
import math
import os
import shutil
from pathlib import Path
from typing import AsyncIterator, Dict, List

from web_deploy.common.uploads import BatchRejected, sanitize_filename

MIN_CHUNK_BYTES = 1024 * 1024
MAX_CHUNK_BYTES = 256 * 1024 * 1024

# One lock per file being finished, so two last pieces arriving at once rename it once
_finishing: Dict[str, asyncio.Lock] = {}


class ChunkedUpload:
    def __init__(self, folder: Path, filename: str, size: int, chunk: int, max_bytes: int) -> None:
        self.name = sanitize_filename(filename)
        if not isinstance(size, int) or size < 0 or size > max_bytes:
            raise BatchRejected("The file is larger than the appliance accepts.")
        if not isinstance(chunk, int) or not MIN_CHUNK_BYTES <= chunk <= MAX_CHUNK_BYTES:
            raise BatchRejected(f"Pieces must be {MIN_CHUNK_BYTES} to {MAX_CHUNK_BYTES} bytes.")
        self.folder = folder
        self.size = size
        self.chunk = chunk
        self.count = max(1, math.ceil(size / chunk))
        self.target = folder / self.name
        self.data = folder / f".{self.name}.chunked"
        self.state = folder / f".{self.name}.pieces"

    # ---------- asking ----------

    def begin(self) -> dict:
        """Readies the file for its pieces (keeping those already here) and says which have arrived."""
        if self._finished():
            return {"received": list(range(self.count)), "complete": True}
        self.folder.mkdir(parents=True, exist_ok=True)
        if self._meta() != {"size": self.size, "chunk": self.chunk}:
            self._reset()
        return {"received": self.received(), "complete": False}

    def received(self) -> List[int]:
        if not self.state.is_dir():
            return []
        return sorted(int(p.name) for p in self.state.iterdir() if p.name.isdigit() and int(p.name) < self.count)

    # ---------- a piece ----------

    def expected_length(self, index: int) -> int:
        if not 0 <= index < self.count:
            raise BatchRejected(f"There is no piece {index}.")
        return min(self.chunk, self.size - index * self.chunk)

    async def write(self, index: int, body: AsyncIterator[bytes]) -> dict:
        """Writes piece `index` at its offset; finishes the file once every piece is here."""
        if self._finished():
            return {"complete": True}
        if self._meta() != {"size": self.size, "chunk": self.chunk}:
            raise BatchRejected("This upload was restarted with other sizes; ask which pieces are here again.")
        expected = self.expected_length(index)
        data = bytearray()
        async for part in body:
            data.extend(part)
            if len(data) > expected:
                raise BatchRejected(f"Piece {index} is longer than {expected} bytes.")
        if len(data) != expected:
            raise BatchRejected(f"Piece {index} arrived incomplete ({len(data)} of {expected} bytes).")
        await asyncio.to_thread(self._write_at, index * self.chunk, bytes(data))
        (self.state / str(index)).touch()
        if len(self.received()) < self.count:
            return {"complete": False}
        lock = _finishing.setdefault(str(self.target), asyncio.Lock())
        async with lock:
            if not self._finished():
                await asyncio.to_thread(self._finish)
        _finishing.pop(str(self.target), None)
        return {"complete": True}

    # ---------- plumbing ----------

    def _write_at(self, offset: int, data: bytes) -> None:
        handle = os.open(self.data, os.O_WRONLY)
        try:
            view = memoryview(data)
            while view:
                written = os.pwrite(handle, view, offset)
                view, offset = view[written:], offset + written
            os.fsync(handle)
        finally:
            os.close(handle)

    def _finish(self) -> None:
        if not self.data.exists():
            return  # finished by the other last piece
        self.data.replace(self.target)
        shutil.rmtree(self.state, ignore_errors=True)

    def _finished(self) -> bool:
        """Done earlier: the file is in place at its full size and nothing of the upload is left."""
        return (self.target.is_file() and self.target.stat().st_size == self.size
                and not self.data.exists() and not self.state.exists())

    def _meta(self) -> dict:
        try:
            return json.loads((self.state / "meta.json").read_text())
        except (OSError, ValueError):
            return {}

    def _reset(self) -> None:
        shutil.rmtree(self.state, ignore_errors=True)
        self.state.mkdir(parents=True)
        with open(self.data, "wb") as handle:
            handle.truncate(self.size)  # sparse: the pieces fill it in
        (self.state / "meta.json").write_text(json.dumps({"size": self.size, "chunk": self.chunk}))


def discard_incomplete(folder: Path) -> int:
    """Removes what unfinished uploads left in a batch (when it is queued): the box must not be sent half files."""
    removed = 0
    if not folder.is_dir():
        return 0
    for path in folder.iterdir():
        if path.name.startswith(".") and (path.name.endswith(".chunked") or path.name.endswith(".part")):
            path.unlink(missing_ok=True)
            removed += 1
        elif path.name.startswith(".") and path.name.endswith(".pieces") and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    return removed
