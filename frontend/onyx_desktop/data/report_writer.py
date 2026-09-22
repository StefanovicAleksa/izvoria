# Canvas: desktop_report_writer_v1
# Path: frontend/onyx_desktop/data/report_writer.py

"""
Saving a finished report to wherever the user chose.

Lives in `data/` because that is where every disk write in this app lives -- the file dialog belongs
to the view, the bytes belong here. No Qt, so it is cheap to unit-test.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from onyx_desktop.domain.entities import ReportDocument

logger = logging.getLogger(__name__)


def default_save_path(report: ReportDocument) -> str:
    """
    An ABSOLUTE default for the save dialog.

    A bare filename resolves against the process CWD, and `apply_runtime_environment` chdirs to the
    project root -- so every saved report landed untracked inside the git working tree. A report is
    client transcript content; it must not default into a repository.
    """
    documents = Path.home() / "Documents"
    folder = documents if documents.is_dir() else Path.home()
    return str(folder / default_filename(report))


def default_filename(report: ReportDocument) -> str:
    """A filesystem-safe suggestion for the save dialog, from the report's own topic."""
    slug = "".join(char if char.isalnum() else "-" for char in report.topic.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")[:60].rstrip("-")
    return f"{slug or 'research-report'}.md"


def save_markdown(report: ReportDocument, destination: str) -> Optional[str]:
    """
    Writes the report to `destination`, returning None on success or a sentence explaining the failure.

    Returns rather than raises, matching `system_open.open_source`: a save that quietly failed is worse
    than one that says so, and a dialog handler should not have to catch.

    Copies the file the engine already wrote when it is still there, so what the user saves is
    byte-identical to what the appliance produced and audited. Falls back to the in-memory Markdown --
    but that carries `onyx://doc/N` links, which are meaningless outside this app, so they are stripped.

    Written atomically: a truncated report looks complete, and the part that gets cut is the references
    table a reader would check.
    """
    target = Path(destination)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)

        source = Path(report.markdown_path) if report.markdown_path else None
        if source is not None and source.exists():
            handle, temp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".report-", suffix=".tmp")
            os.close(handle)
            try:
                shutil.copyfile(source, temp_path)
                os.replace(temp_path, target)
            except BaseException:
                Path(temp_path).unlink(missing_ok=True)
                raise
            logger.info(f"Saved report {report.report_id} to {target}")
            return None

        handle, temp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".report-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(_plain_citations(report.markdown))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, target)
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise

        logger.info(f"Saved report {report.report_id} to {target} (re-rendered; the engine's file is gone)")
        return None

    except OSError as exc:
        logger.error(f"Could not save report {report.report_id} to {target}: {exc}")
        return f"Could not write that file:\n{exc.strerror or exc}"


def _plain_citations(markdown: str) -> str:
    """
    Turns `[Doc 7](onyx://doc/7)` back into `[Doc 7]`.

    The link scheme only means something inside this app. A saved file has to be readable on its own --
    that is the entire point of the references table.
    """
    import re
    return re.sub(r"\[(Doc \d+)]\(onyx://doc/\d+\)", r"[\1]", markdown)
