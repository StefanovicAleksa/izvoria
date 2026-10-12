"""
The ingestion wait list, shared by every firm on the box.

Ingestion needs most of the ingestion half of the GPU for minutes to hours, so batches run one at a time, first come
first served, whichever firm they belong to. The list lives in its own small database (`onyx_control`), not in any
firm's: the API enqueues and lists, the supervisor claims and records, and both survive each other's restarts.

This is ops state, not appliance schema, so the table is created with `create_all` rather than an Alembic revision.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import (
    JSON, Column, DateTime, Integer, MetaData, String, Table, Text, create_engine, func, inspect, select, text, update,
)
from sqlalchemy.engine import Engine


class JobStatus(str, Enum):
    # The VPS is making the batch's 720p copies (web_deploy/gateway/prepare.py); the box never claims it yet
    PREPARING = "preparing"
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IngestMode(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    IMAGE = "image"


class JobOrigin(str, Enum):
    """Where a batch's files come from: uploaded by the user, or downloaded from YouTube before it runs."""
    UPLOAD = "upload"
    YOUTUBE = "youtube"


metadata = MetaData()

ingest_jobs = Table(
    "ingest_jobs", metadata,
    Column("id", String(36), primary_key=True),
    Column("firm", String(64), nullable=False, index=True),
    Column("owner_id", String(64), nullable=False),
    Column("mode", String(16), nullable=False),
    Column("language", String(8), nullable=False),
    Column("batch_dir", Text, nullable=False),
    Column("file_count", Integer, nullable=False, default=0),
    Column("status", String(16), nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("progress", JSON, nullable=True),
    Column("report", JSON, nullable=True),
    Column("error", Text, nullable=True),
    Column("origin", String(16), nullable=False, default=JobOrigin.UPLOAD.value,
           server_default=JobOrigin.UPLOAD.value),
    # A YouTube batch's videos (youtube.write_manifest's content). Kept in the row, not the batch folder: the folder
    # lives on the VPS, and the GPU box that resolved the links never writes there
    Column("manifest", JSON, nullable=True),
)

# Columns added after the table first shipped, as (name, DDL). `create_all` makes a missing table but never alters an
# existing one, so `connect` adds these to a queue database from before them.
_ADDED_COLUMNS = (
    ("origin", f"VARCHAR(16) NOT NULL DEFAULT '{JobOrigin.UPLOAD.value}'"),
    ("manifest", "JSON"),
)


class QueueError(ValueError):
    pass


@dataclass(frozen=True)
class IngestJob:
    id: str
    firm: str
    owner_id: str
    mode: IngestMode
    language: str
    batch_dir: str
    file_count: int
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    progress: Optional[dict]
    report: Optional[dict]
    error: Optional[str]
    origin: JobOrigin = JobOrigin.UPLOAD
    manifest: Optional[dict] = None

    def public(self, position: Optional[int] = None) -> dict:
        """What the firm's users see. Never the box path."""
        return {
            "id": self.id, "mode": self.mode.value, "language": self.language, "file_count": self.file_count,
            "status": self.status.value, "position": position, "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "progress": self.progress, "report": self.report, "error": self.error, "origin": self.origin.value,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite (the local queue) drops the timezone it was given; every time here was written in UTC."""
    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


def _job(row: Any) -> IngestJob:
    m = row._mapping
    return IngestJob(
        id=m["id"], firm=m["firm"], owner_id=m["owner_id"], mode=IngestMode(m["mode"]), language=m["language"],
        batch_dir=m["batch_dir"], file_count=m["file_count"] or 0, status=JobStatus(m["status"]),
        created_at=_utc(m["created_at"]), started_at=_utc(m["started_at"]), finished_at=_utc(m["finished_at"]),
        progress=m["progress"], report=m["report"], error=m["error"],
        origin=JobOrigin(m["origin"] or JobOrigin.UPLOAD.value), manifest=m["manifest"],
    )


def _add_missing_columns(engine: Engine) -> None:
    present = {column["name"] for column in inspect(engine).get_columns(ingest_jobs.name)}
    for name, ddl in _ADDED_COLUMNS:
        if name not in present:
            with engine.begin() as connection:
                connection.execute(text(f"ALTER TABLE {ingest_jobs.name} ADD COLUMN {name} {ddl}"))


class IngestQueue:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def connect(cls, url: str) -> "IngestQueue":
        # A bare postgresql:// (e.g. CONTROL_DATABASE_URL set by hand) means psycopg 3 under SQLAlchemy 2.1+, which
        # isn't installed: name the driver the box has
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url[len("postgresql://"):]
        if url.startswith("sqlite"):
            # Unit tests: one shared in-memory database across threads
            from sqlalchemy.pool import StaticPool
            engine = create_engine(url, poolclass=StaticPool, connect_args={"check_same_thread": False})
        else:
            engine = create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=2)
        queue = cls(engine)
        metadata.create_all(engine)
        _add_missing_columns(engine)
        return queue

    # ---------- API side ----------

    def enqueue(self, firm: str, owner_id: str, mode: IngestMode, language: str, batch_dir: str,
                file_count: int, origin: JobOrigin = JobOrigin.UPLOAD,
                manifest: Optional[dict] = None, prepare: bool = False) -> IngestJob:
        if file_count <= 0:
            raise QueueError("The batch has no files.")
        job_id = str(uuid.uuid4())
        with self._engine.begin() as connection:
            connection.execute(ingest_jobs.insert().values(
                id=job_id, firm=firm, owner_id=owner_id, mode=mode.value, language=language, batch_dir=batch_dir,
                file_count=file_count, status=(JobStatus.PREPARING if prepare else JobStatus.QUEUED).value,
                created_at=_now(), origin=origin.value, manifest=manifest,
            ))
        return self.get(job_id)

    def get(self, job_id: str) -> Optional[IngestJob]:
        with self._engine.connect() as connection:
            row = connection.execute(select(ingest_jobs).where(ingest_jobs.c.id == job_id)).first()
        return _job(row) if row else None

    def list_for_firm(self, firm: str, limit: int = 50) -> List[dict]:
        """The firm's jobs, newest first, each queued one with its place in the whole box's line (1 = next)."""
        positions = self.queue_positions()
        with self._engine.connect() as connection:
            rows = connection.execute(
                select(ingest_jobs).where(ingest_jobs.c.firm == firm)
                .order_by(ingest_jobs.c.created_at.desc()).limit(limit)
            ).all()
        return [_job(r).public(positions.get(r._mapping["id"])) for r in rows]

    def queue_positions(self) -> Dict[str, int]:
        with self._engine.connect() as connection:
            ids = connection.execute(
                select(ingest_jobs.c.id).where(ingest_jobs.c.status == JobStatus.QUEUED.value)
                .order_by(ingest_jobs.c.created_at, ingest_jobs.c.id)
            ).scalars().all()
        return {job_id: index for index, job_id in enumerate(ids, start=1)}

    def cancel(self, job_id: str, firm: str) -> bool:
        """Takes a job off the list if it hasn't started. A running ingestion is not interruptible."""
        with self._engine.begin() as connection:
            result = connection.execute(
                update(ingest_jobs)
                .where(ingest_jobs.c.id == job_id, ingest_jobs.c.firm == firm,
                       ingest_jobs.c.status.in_([JobStatus.PREPARING.value, JobStatus.QUEUED.value]))
                .values(status=JobStatus.CANCELLED.value, finished_at=_now())
            )
            return bool(result.rowcount)

    def depth(self) -> Dict[str, int]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                select(ingest_jobs.c.status, func.count()).where(
                    ingest_jobs.c.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value])
                ).group_by(ingest_jobs.c.status)
            ).all()
        counts = {status: count for status, count in rows}
        return {"queued": counts.get(JobStatus.QUEUED.value, 0), "running": counts.get(JobStatus.RUNNING.value, 0)}

    # ---------- supervisor side ----------

    def last_finished_at(self) -> Optional[datetime]:
        """When the most recent job ended (done, failed or cancelled): the backup runs after new material lands."""
        with self._engine.connect() as connection:
            value = connection.execute(select(func.max(ingest_jobs.c.finished_at))).scalar()
        return _utc(value) if value else None

    def active_batch_dirs(self) -> Set[str]:
        """The upload folders a job will still read (being prepared, queued or running): a sweep leaves them alone."""
        with self._engine.connect() as connection:
            return set(connection.execute(
                select(ingest_jobs.c.batch_dir).where(ingest_jobs.c.status.in_(
                    [JobStatus.PREPARING.value, JobStatus.QUEUED.value, JobStatus.RUNNING.value]))
            ).scalars().all())

    # ---------- the VPS's video preparer ----------

    def next_to_prepare(self) -> Optional[IngestJob]:
        """The oldest job whose videos still need their 720p copies. One preparer runs, so nothing is claimed."""
        with self._engine.connect() as connection:
            job_id = connection.execute(
                select(ingest_jobs.c.id).where(ingest_jobs.c.status == JobStatus.PREPARING.value)
                .order_by(ingest_jobs.c.created_at, ingest_jobs.c.id).limit(1)
            ).scalar()
        return self.get(job_id) if job_id else None

    def mark_prepared(self, job_id: str, file_count: Optional[int] = None) -> bool:
        """Hands a prepared job to the box's line, unless it was cancelled meanwhile. Returns whether it did."""
        values: Dict[str, Any] = {"status": JobStatus.QUEUED.value, "progress": None}
        if file_count is not None:
            values["file_count"] = file_count
        with self._engine.begin() as connection:
            return bool(connection.execute(
                update(ingest_jobs)
                .where(ingest_jobs.c.id == job_id, ingest_jobs.c.status == JobStatus.PREPARING.value)
                .values(**values)
            ).rowcount)

    def finished_batch_dirs(self) -> Set[str]:
        """The upload folders of jobs that can never run again (done, failed or cancelled): safe to remove."""
        with self._engine.connect() as connection:
            return set(connection.execute(
                select(ingest_jobs.c.batch_dir).where(ingest_jobs.c.status.in_(
                    [JobStatus.DONE.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]))
            ).scalars().all())

    def claim_next(self) -> Optional[IngestJob]:
        """
        Marks the oldest queued job running and returns it. The claim is a conditional update, so even two
        supervisors could never run the same job: only one update finds it still queued.
        """
        while True:
            with self._engine.begin() as connection:
                job_id = connection.execute(
                    select(ingest_jobs.c.id).where(ingest_jobs.c.status == JobStatus.QUEUED.value)
                    .order_by(ingest_jobs.c.created_at, ingest_jobs.c.id).limit(1)
                ).scalar()
                if job_id is None:
                    return None
                claimed = connection.execute(
                    update(ingest_jobs)
                    .where(ingest_jobs.c.id == job_id, ingest_jobs.c.status == JobStatus.QUEUED.value)
                    .values(status=JobStatus.RUNNING.value, started_at=_now())
                ).rowcount
            if claimed:
                return self.get(job_id)

    def record_progress(self, job_id: str, progress: dict) -> None:
        with self._engine.begin() as connection:
            connection.execute(update(ingest_jobs).where(ingest_jobs.c.id == job_id).values(progress=progress))

    def finish(self, job_id: str, report: dict) -> None:
        with self._engine.begin() as connection:
            connection.execute(update(ingest_jobs).where(ingest_jobs.c.id == job_id).values(
                status=JobStatus.DONE.value, finished_at=_now(), report=report))

    def fail(self, job_id: str, error: str) -> None:
        with self._engine.begin() as connection:
            connection.execute(update(ingest_jobs).where(ingest_jobs.c.id == job_id).values(
                status=JobStatus.FAILED.value, finished_at=_now(), error=error[:4000]))

    def recover_interrupted(self) -> int:
        """
        Jobs left `running` by a crash or a box stop. Their partial work is cached per source in the firm's database,
        so they are re-queued at the front rather than failed: the retry skips what was already done.
        """
        with self._engine.begin() as connection:
            return connection.execute(
                update(ingest_jobs).where(ingest_jobs.c.status == JobStatus.RUNNING.value)
                .values(status=JobStatus.QUEUED.value, started_at=None,
                        progress={"phase": "interrupted", "state": "requeued", "detail": None})
            ).rowcount
