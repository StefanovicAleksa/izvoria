# Path: frontend/tests/data/test_report_writer_unit.py

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from onyx_desktop.data.report_writer import default_filename, save_markdown
from onyx_desktop.domain.entities import ReportDocument
from onyx_desktop.domain.enums import ReportRunStatus

REPORT_ID = uuid.UUID("55555555-5555-5555-5555-555555555555")

_LINKED = "# Topic\n\nSealed [Doc 1](onyx://doc/1) and stored [Doc 3](onyx://doc/3).\n"


def _document(markdown_path=None, markdown=_LINKED) -> ReportDocument:
    return ReportDocument(
        report_id=REPORT_ID,
        topic="Chain of custody: the Ramirez matter",
        markdown=markdown,
        status=ReportRunStatus.COMPLETED,
        markdown_path=str(markdown_path) if markdown_path else None,
    )


class TestSaveMarkdown:
    def test_copies_the_engines_own_file_byte_for_byte(self, tmp_path):
        """
        What the user saves must be identical to what the appliance produced and audited -- not a
        re-render that could drift from it.
        """
        source = tmp_path / "engine.md"
        source.write_text("# Exactly this\n", encoding="utf-8")
        destination = tmp_path / "saved.md"

        assert save_markdown(_document(markdown_path=source), str(destination)) is None
        assert destination.read_text(encoding="utf-8") == "# Exactly this\n"

    def test_falls_back_to_the_rendered_markdown_when_the_file_is_gone(self, tmp_path):
        destination = tmp_path / "saved.md"

        assert save_markdown(_document(markdown_path=tmp_path / "missing.md"), str(destination)) is None
        assert "Sealed [Doc 1]" in destination.read_text(encoding="utf-8")

    def test_the_fallback_strips_the_internal_link_scheme(self, tmp_path):
        """
        `onyx://doc/N` only means something inside this app. A saved report has to be readable on its
        own -- that is the entire point of the references table.
        """
        destination = tmp_path / "saved.md"

        save_markdown(_document(), str(destination))

        written = destination.read_text(encoding="utf-8")
        assert "onyx://" not in written
        assert "[Doc 1]" in written and "[Doc 3]" in written

    def test_returns_an_explanation_instead_of_raising(self, tmp_path):
        """Matches system_open: a save that quietly failed is worse than one that says so."""
        destination = tmp_path / "saved.md"

        with patch("os.replace", side_effect=OSError(28, "No space left on device")):
            error = save_markdown(_document(), str(destination))

        assert error is not None
        assert "No space left on device" in error

    def test_a_failed_save_leaves_no_partial_file(self, tmp_path):
        """A truncated report looks complete, and the part cut off is the references table."""
        destination = tmp_path / "saved.md"

        with patch("os.replace", side_effect=OSError(28, "No space left on device")):
            save_markdown(_document(), str(destination))

        assert not destination.exists()
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []

    def test_missing_parent_directories_are_created(self, tmp_path):
        destination = tmp_path / "nested" / "deeper" / "saved.md"

        assert save_markdown(_document(), str(destination)) is None
        assert destination.exists()

    def test_overwrites_an_existing_file(self, tmp_path):
        destination = tmp_path / "saved.md"
        destination.write_text("stale", encoding="utf-8")

        assert save_markdown(_document(), str(destination)) is None
        assert "stale" not in destination.read_text(encoding="utf-8")


@pytest.mark.parametrize("topic,expected", [
    ("Chain of custody", "chain-of-custody.md"),
    ("Chain of custody: the Ramirez matter!", "chain-of-custody-the-ramirez-matter.md"),
    ("   ", "research-report.md"),
    ("!!!", "research-report.md"),
])
def test_default_filename(topic, expected):
    document = ReportDocument(
        report_id=REPORT_ID, topic=topic, markdown="", status=ReportRunStatus.COMPLETED
    )

    assert default_filename(document) == expected


class TestDefaultSavePath:
    def test_it_is_absolute_and_outside_the_repository(self):
        """
        A bare filename resolves against the CWD, and the app chdirs to the project root -- saved
        reports were landing untracked in the git working tree. A report is client transcript content.
        """
        from pathlib import Path

        from onyx_desktop.data.report_writer import default_save_path

        document = ReportDocument(
            report_id=REPORT_ID, topic="Chain of custody", markdown="",
            status=ReportRunStatus.COMPLETED,
        )

        path = Path(default_save_path(document))

        assert path.is_absolute()
        assert path.name == "chain-of-custody.md"
        assert str(path).startswith(str(Path.home()))
