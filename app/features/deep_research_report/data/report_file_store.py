# Path: app/features/deep_research_report/data/report_file_store.py

import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.features.deep_research_report.domain.interfaces import IReportFileStore
from app.features.deep_research_report.utils.error_mapper import map_report_error

logger = logging.getLogger(__name__)


class ReportFileStore(IReportFileStore):
    """
    Writes finished reports under `settings.REPORTS_DIR`, date-sharded the way the media vault is
    (`FileManager.generate_path`), so a year of reports doesn't end up in one flat directory.

    Unlike the vault, files are left writable: a resumed or re-rendered report overwrites its own file, and
    the vault's read-only 0o644-after-store pattern would make the second write fail.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        # Resolved per call rather than captured here, so tests that repoint settings.REPORTS_PATH take
        # effect without rebuilding the store.
        self._base_dir = base_dir

    def _resolve_dir(self) -> Path:
        now = datetime.now(timezone.utc)
        root = self._base_dir if self._base_dir is not None else settings.REPORTS_DIR
        target = Path(root) / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_markdown(self, report_id: uuid.UUID, markdown: str) -> str:
        """
        Writes atomically: a temp file in the destination directory, then `os.replace`.

        A half-written report is worse than a missing one -- it looks complete, and the part that got
        truncated is exactly the references table a reader would check.
        """
        try:
            target_dir = self._resolve_dir()
            destination = target_dir / f"{report_id}.md"

            handle, temp_path = tempfile.mkstemp(
                dir=str(target_dir), prefix=f".{report_id}.", suffix=".md.tmp"
            )
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    stream.write(markdown)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_path, destination)
            except Exception:
                # Never leave the dotfile behind; the next write would not reuse it and nothing else
                # would ever clean it up.
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

            logger.info(f"Wrote research report {report_id} to {destination}")
            return str(destination)

        except Exception as e:
            raise map_report_error(e, context=f"Writing report {report_id} to disk") from e
