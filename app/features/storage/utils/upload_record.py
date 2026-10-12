# Path: app/features/storage/utils/upload_record.py
"""
What a file was before it was prepared for ingestion, when it no longer is the bytes that were uploaded.

The hosted version re-encodes uploaded videos to 720p on its VPS before the GPU box ingests them
(web_deploy/storage/prepare.py). A source is deduplicated by the hash of what was uploaded, and ffmpeg's output is
not reproducible, so hashing the copy would make the same upload a new source every time. The preparer therefore
leaves a record beside the copies: for each prepared file, the original's sha256 and name. Without the record (the
appliance, every other upload) a file is hashed as always.

Standard library only: the gateway writes this record with the same module.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

RECORD_NAME = ".upload-record.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class UploadRecord:
    sha256: str
    original_name: str


def read_record(path: Path) -> Optional[UploadRecord]:
    """The record of the file at `path`, from its folder's record file; None when it has none (or it is unusable)."""
    try:
        entries = json.loads((Path(path).parent / RECORD_NAME).read_text(encoding="utf-8")).get("files", {})
        entry = entries.get(Path(path).name)
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    if not isinstance(entry, dict):
        return None
    sha256, name = entry.get("sha256"), entry.get("original_name")
    if not isinstance(sha256, str) or not _SHA256.match(sha256) or not isinstance(name, str) or not name:
        return None
    return UploadRecord(sha256=sha256, original_name=name)


def write_record(folder: Path, records: Dict[str, UploadRecord]) -> None:
    """Records, keyed by the prepared file's name. Written atomically; replaces the folder's earlier record."""
    target = Path(folder) / RECORD_NAME
    partial = Path(folder) / f"{RECORD_NAME}.part"
    partial.write_text(json.dumps({"files": {name: {"sha256": r.sha256, "original_name": r.original_name}
                                             for name, r in sorted(records.items())}}, indent=1), encoding="utf-8")
    partial.replace(target)


def read_all(folder: Path) -> Dict[str, UploadRecord]:
    """Every usable record in a folder."""
    try:
        names = json.loads((Path(folder) / RECORD_NAME).read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError, AttributeError, TypeError):
        return {}
    found = {}
    for name in names if isinstance(names, dict) else {}:
        record = read_record(Path(folder) / name)
        if record is not None:
            found[name] = record
    return found
