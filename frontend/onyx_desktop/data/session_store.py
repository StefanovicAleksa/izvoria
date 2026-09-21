# Canvas: desktop_session_store_v1
# Path: frontend/onyx_desktop/data/session_store.py

"""
A GUI-local registry of conversations.

The engine stores every turn in `session_turns` but has no way to *list* sessions: a session id
is minted by the caller and a conversation is only retrievable if you already hold its UUID. So
the GUI keeps its own index -- id, title, timestamp, turn count -- and the engine remains the
source of truth for turn content.

Titles are derived from the first question; nothing upstream stores one.
"""

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from onyx_desktop.domain.entities import SessionRef
from onyx_desktop.utils.formatting import elide

logger = logging.getLogger(__name__)

_MAX_SESSIONS = 200


def default_store_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "onyx-desktop" / "sessions.json"


class SessionStore:
    """
    Reads and writes the conversation index. Every failure degrades to "no history" rather than
    breaking the window: a missing or corrupt index is an inconvenience, not a fault.
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path is not None else default_store_path()

    @property
    def path(self) -> Path:
        return self._path

    def list_sessions(self) -> List[SessionRef]:
        """Newest first."""
        records = self._read()
        sessions: List[SessionRef] = []
        for record in records:
            try:
                sessions.append(SessionRef(
                    session_id=uuid.UUID(record["session_id"]),
                    title=record.get("title") or "Untitled conversation",
                    started_at=datetime.fromisoformat(record["started_at"]),
                    turn_count=int(record.get("turn_count", 0)),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(f"Skipping malformed session record: {exc}")
        sessions.sort(key=lambda s: s.started_at, reverse=True)
        return sessions

    def remember_turn(self, session_id: uuid.UUID, question: str) -> SessionRef:
        """
        Records a session on its first question and bumps the turn count after that. Returns the
        stored reference so the sidebar can update without re-reading the file.
        """
        records = self._read()
        key = str(session_id)
        for record in records:
            if record.get("session_id") == key:
                record["turn_count"] = int(record.get("turn_count", 0)) + 1
                self._write(records)
                return SessionRef(
                    session_id=session_id,
                    title=record.get("title") or "Untitled conversation",
                    started_at=datetime.fromisoformat(record["started_at"]),
                    turn_count=record["turn_count"],
                )

        started = datetime.now(timezone.utc)
        title = elide(question, 60) or "Untitled conversation"
        records.append({
            "session_id": key,
            "title": title,
            "started_at": started.isoformat(),
            "turn_count": 1,
        })
        self._write(records[-_MAX_SESSIONS:])
        return SessionRef(session_id=session_id, title=title, started_at=started, turn_count=1)

    def forget(self, session_id: uuid.UUID) -> None:
        key = str(session_id)
        self._write([r for r in self._read() if r.get("session_id") != key])

    # ---------- file handling ----------

    def _read(self) -> List[dict]:
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not read the session index at {self._path}: {exc}")
            return []
        return data if isinstance(data, list) else []

    def _write(self, records: List[dict]) -> None:
        """
        Writes atomically: a crash mid-write would otherwise leave a truncated index that reads
        back as "no conversations".
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(self._path.parent),
                prefix=".sessions-", suffix=".tmp", delete=False,
            )
            try:
                with handle:
                    json.dump(records, handle, indent=2)
                os.replace(handle.name, self._path)
            except BaseException:
                Path(handle.name).unlink(missing_ok=True)
                raise
        except OSError as exc:
            logger.warning(f"Could not write the session index at {self._path}: {exc}")
