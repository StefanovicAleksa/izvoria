"""
Removing ingested files from a firm's index, with everything ingestion made from them.

Two callers:
- the gateway on the VPS, when a firm administrator deletes a file from the Library (it holds the firm databases and
  the vaults, so this works while the GPU box is stopped);
- the box's ingestion supervisor, when a running batch is stopped: the file that was half-way through is removed, so
  it neither lingers without chunks in the Library nor keeps the batch's language stamped on it.

Almost every table built from a source has ON DELETE CASCADE from `sources`. What the cascade misses is handled
here: `jobs.source_id` has no cascade (it would refuse the delete), and the extracted files (WAV, clips, frames) are
`files` rows of their own that nothing would delete. The vault files themselves are unlinked by the caller with
`unlink_inside`, after the rows are gone.

Standard library and a DB-API connection (psycopg2) only: imported by the gateway and the box supervisor alike.
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)

# Every vault file of the given sources: the original, its extracted audio, per-chunk clips and the frames visual
# analysis read (file ids inside extracted_chunk_frames.frames). Returns (files.id, files.file_path).
SOURCE_FILES_SQL = """
WITH matched AS (SELECT id, file_id FROM sources WHERE id = ANY(%(source_ids)s::uuid[]))
SELECT f.id, f.file_path FROM files f WHERE f.id IN (
    SELECT file_id FROM matched
    UNION SELECT extracted_file_id FROM audio_extractions WHERE source_id IN (SELECT id FROM matched)
    UNION SELECT extracted_file_id FROM video_clips WHERE source_id IN (SELECT id FROM matched)
    UNION SELECT (frame ->> 'file_id')::uuid
          FROM extracted_chunk_frames e, jsonb_array_elements(
              CASE WHEN jsonb_typeof(e.frames) = 'array' THEN e.frames ELSE '[]'::jsonb END) AS frame
          WHERE e.source_id IN (SELECT id FROM matched) AND frame ? 'file_id'
)
"""

SOURCES_BY_HASH_SQL = "SELECT id FROM sources WHERE content_hash = ANY(%(hashes)s)"

# A source of a stopped batch that never reached the index: created while the batch ran, and without a single chunk
UNFINISHED_SOURCES_SQL = """
SELECT s.id FROM sources s
WHERE s.created_at >= %(since)s
  AND NOT EXISTS (SELECT 1 FROM vector_chunks c WHERE c.source_id = s.id)
"""

# (firm) -> an open DB-API connection to that firm's database
Connect = Callable[[str], object]


def _ids(source_ids: Iterable) -> List[str]:
    return [str(uuid.UUID(str(s))) for s in source_ids]


def source_files(connection, source_ids: Sequence) -> List[Tuple[str, str]]:
    """(file id, absolute path) of every vault file the sources own."""
    ids = _ids(source_ids)
    if not ids:
        return []
    with connection.cursor() as cursor:
        cursor.execute(SOURCE_FILES_SQL, {"source_ids": ids})
        return [(str(row[0]), row[1]) for row in cursor.fetchall()]


def sources_with_hashes(connection, hashes: Sequence[str]) -> List[str]:
    if not hashes:
        return []
    with connection.cursor() as cursor:
        cursor.execute(SOURCES_BY_HASH_SQL, {"hashes": list(hashes)})
        return [str(row[0]) for row in cursor.fetchall()]


def existing_sources(connection, source_ids: Sequence) -> List[str]:
    ids = _ids(source_ids)
    if not ids:
        return []
    with connection.cursor() as cursor:
        cursor.execute("SELECT id FROM sources WHERE id = ANY(%(source_ids)s::uuid[])", {"source_ids": ids})
        return [str(row[0]) for row in cursor.fetchall()]


def unfinished_sources(connection, since: datetime) -> List[str]:
    with connection.cursor() as cursor:
        cursor.execute(UNFINISHED_SOURCES_SQL, {"since": since})
        return [str(row[0]) for row in cursor.fetchall()]


def remove_sources(connection, source_ids: Sequence) -> List[str]:
    """
    Deletes the sources and every row made from them, in one transaction (committed here). Returns the vault paths
    their files had, for the caller to unlink once the rows are gone.
    """
    ids = _ids(source_ids)
    if not ids:
        return []
    files = source_files(connection, ids)
    file_ids = [file_id for file_id, _ in files]
    try:
        with connection.cursor() as cursor:
            # The ingestion bookkeeping points at the source without a cascade
            cursor.execute("DELETE FROM jobs WHERE source_id = ANY(%(source_ids)s::uuid[])", {"source_ids": ids})
            # Every derived row goes with the source (ON DELETE CASCADE), the audio and clip link rows included
            cursor.execute("DELETE FROM sources WHERE id = ANY(%(source_ids)s::uuid[])", {"source_ids": ids})
            if file_ids:
                # The original and the extracted files, which nothing references any more
                cursor.execute("DELETE FROM files WHERE id = ANY(%(file_ids)s::uuid[])", {"file_ids": file_ids})
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return [path for _, path in files]


def unlink_inside(paths: Iterable[str], vault_root: Path) -> int:
    """
    Deletes each file that lies inside the vault (never through a symlink, never outside it), then the folders left
    empty, up to the vault itself. Returns how many files were removed.
    """
    root = Path(vault_root).resolve()
    removed = 0
    for raw in paths:
        path = Path(raw)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if root not in resolved.parents or path.is_symlink():
            logger.warning(f"Not removing {raw}: it is not a file inside {root}.")
            continue
        try:
            resolved.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(f"Could not remove {raw}: {exc}")
            continue
        parent = resolved.parent
        while parent != root and root in parent.parents:
            try:
                parent.rmdir()  # only when empty
            except OSError:
                break
            parent = parent.parent
    return removed


def postgres_connector(env=None, password: Callable[[], object] = lambda: None) -> Connect:
    """
    Opens a firm's database with the POSTGRES_* settings. `password` overrides POSTGRES_PASSWORD per connection (the
    GPU box asks its gateway for the current one; see web_deploy/common/box_credentials.py).
    """
    from web_deploy.common.tenancy import database_name
    settings = dict(os.environ if env is None else env)

    def connect(firm: str):
        import psycopg2
        return psycopg2.connect(host=settings.get("POSTGRES_SERVER", "127.0.0.1"),
                                port=settings.get("POSTGRES_PORT", "5432"),
                                user=settings.get("POSTGRES_USER", "onyx"),
                                password=password() or settings.get("POSTGRES_PASSWORD", ""),
                                dbname=database_name(firm), connect_timeout=15)
    return connect
