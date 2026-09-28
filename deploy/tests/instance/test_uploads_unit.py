import ast
from pathlib import Path

import pytest

from deploy.instance.agent.uploads import ALLOWED_SUFFIXES, UploadRejected, sanitize_filename, upload_path

REPO_ROOT = Path(__file__).resolve().parents[3]
BATCH = "20260928-153000"


class TestSanitizeFilename:
    @pytest.mark.parametrize("raw, expected", [
        ("hearing.mp4", "hearing.mp4"),
        ("Saslušanje 12.MP4", "Saslušanje 12.MP4"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("/abs/path/report.docx", "report.docx"),
        ("C:\\Users\\x\\scan.pdf", "scan.pdf"),
        ("...hidden.txt", "hidden.txt"),
        ("a;rm -rf$(x).wav", "a_rm -rf__x_.wav"),
    ])
    def test_keeps_only_a_safe_basename(self, raw, expected):
        assert sanitize_filename(raw) == expected

    @pytest.mark.parametrize("raw", ["noext", "evil.sh", "archive.zip", "", "../", "photo.heic"])
    def test_rejects_what_the_appliance_cannot_ingest(self, raw):
        with pytest.raises(UploadRejected):
            sanitize_filename(raw)

    def test_truncates_long_names_but_keeps_the_suffix(self):
        name = sanitize_filename("x" * 400 + ".mp4")
        assert len(name) == 180 and name.endswith(".mp4")


class TestUploadPath:
    def test_places_the_file_under_language_and_batch(self, tmp_path):
        assert upload_path(tmp_path, "sr", BATCH, "clip.mp4") == (tmp_path / "sr" / BATCH / "clip.mp4").resolve()

    @pytest.mark.parametrize("batch", ["..", "../sr", "2026", "20260928-153000/..", ""])
    def test_rejects_batches_that_are_not_timestamps(self, tmp_path, batch):
        with pytest.raises(UploadRejected):
            upload_path(tmp_path, "sr", batch, "clip.mp4")


class TestSuffixParity:
    def test_matches_the_scanners_allowed_extensions(self):
        """The agent can't import app.* (separate venv), so read the scanner's sets straight from its source."""
        tree = ast.parse((REPO_ROOT / "app/features/scanner/utils/filters.py").read_text())
        sets = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Set):
                sets[node.targets[0].id] = {elt.value for elt in node.value.elts}
        scanner = sets["VIDEO_EXTENSIONS"] | sets["AUDIO_EXTENSIONS"] | sets["TEXT_EXTENSIONS"] | sets["IMAGE_EXTENSIONS"]
        assert ALLOWED_SUFFIXES == scanner
