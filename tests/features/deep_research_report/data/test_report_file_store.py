# Path: tests/features/deep_research_report/data/test_report_file_store.py

import os
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from app.features.deep_research_report.data.report_file_store import ReportFileStore
from app.features.deep_research_report.exceptions.storage import ReportFileError


@pytest.fixture
def store(tmp_path):
    return ReportFileStore(base_dir=tmp_path)


class TestWriteMarkdown:
    def test_writes_the_report_and_returns_its_path(self, store, tmp_path):
        report_id = uuid.uuid4()

        path = store.write_markdown(report_id, "# Report\n")

        assert Path(path).read_text(encoding="utf-8") == "# Report\n"
        assert Path(path).name == f"{report_id}.md"

    def test_files_are_date_sharded_like_the_media_vault(self, store, tmp_path):
        path = Path(store.write_markdown(uuid.uuid4(), "# R\n"))

        relative = path.relative_to(tmp_path)
        assert len(relative.parts) == 4  # YYYY / MM / DD / <id>.md
        assert all(part.isdigit() for part in relative.parts[:3])

    def test_a_resume_rewrites_the_same_file_in_place(self, store):
        """
        Unlike the vault, report files stay writable: a resumed run re-renders its own report, and the
        vault's read-only-after-store pattern would make the second write fail.
        """
        report_id = uuid.uuid4()

        first = store.write_markdown(report_id, "# First\n")
        second = store.write_markdown(report_id, "# Second\n")

        assert first == second
        assert Path(second).read_text(encoding="utf-8") == "# Second\n"

    def test_no_temp_files_are_left_behind(self, store):
        path = Path(store.write_markdown(uuid.uuid4(), "# R\n"))

        leftovers = [p.name for p in path.parent.iterdir() if p.name.startswith(".")]
        assert leftovers == []

    def test_a_failed_write_leaves_no_partial_file_and_raises_a_typed_error(self, store, tmp_path):
        """
        A half-written report is worse than a missing one: it looks complete, and the part that got
        truncated is exactly the references table a reader would check.
        """
        report_id = uuid.uuid4()

        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(ReportFileError):
                store.write_markdown(report_id, "# R\n")

        written = list(tmp_path.rglob("*"))
        assert not [p for p in written if p.is_file()]

    def test_unicode_survives_the_round_trip(self, store):
        markdown = "# Chain of custody\n\n00:12:31–00:13:04 — SPEAKER_01 said “sealed”.\n"

        path = store.write_markdown(uuid.uuid4(), markdown)

        assert Path(path).read_text(encoding="utf-8") == markdown


def test_defaults_to_the_configured_reports_directory(tmp_path, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "REPORTS_PATH", str(tmp_path / "configured"))

    path = ReportFileStore().write_markdown(uuid.uuid4(), "# R\n")

    assert str(tmp_path / "configured") in path
