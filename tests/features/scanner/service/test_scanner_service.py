import pytest
from unittest.mock import patch
from pathlib import Path
from app.features.scanner.service.scanner import FileScanner
from app.features.scanner import exceptions as exc


class TestFileScannerService:

    @pytest.fixture
    def scanner(self):
        return FileScanner()

    def test_scan_root_not_found(self, scanner):
        """Scenario: User provides a path that does not exist."""
        # We mock Path.exists specifically, NOT the Path class itself
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(exc.ScanPathNotFound):
                list(scanner.scan("/ghost/dir"))

    def test_scan_happy_path(self, scanner):
        """
        Scenario:
        - Root exists.
        - Contains 1 valid file (video.mp4).
        - Contains 1 invalid file (script.py).
        """
        # 1. Mock the File System Checks (exists, is_dir, stat)
        # We let the Service create REAL Path objects, but we intercept the disk calls.
        with patch("pathlib.Path.exists", return_value=True), \
                patch("pathlib.Path.is_dir", return_value=True), \
                patch("pathlib.Path.stat") as mock_stat:
            # Configure stat to return a fake size
            mock_stat.return_value.st_size = 1024 * 1024  # 1MB

            # 2. Mock os.walk
            # os.walk yields tuples: (root, dirs, files)
            # We simulate one folder with two files
            mock_walk_data = [
                ("/root", [], ["video.mp4", "script.py"])
            ]

            with patch("app.features.scanner.service.scanner.os.walk", return_value=mock_walk_data):
                # 3. Execute
                results = list(scanner.scan("/root"))

                # 4. Assertions
                assert len(results) == 1

                # Verify the properties
                result_file = results[0]
                assert result_file.file_name == "video.mp4"
                assert result_file.extension == ".mp4"
                assert result_file.size_bytes == 1048576

                # Verify Pydantic accepted the path (It should be a real Path object now)
                assert isinstance(result_file.path, Path)
                # Note: On Windows this might normalize differently, but for unit tests strings match usually
                assert str(result_file.path).endswith("video.mp4")

    def test_scan_skips_ignored_directories(self, scanner):
        """
        Scenario: Ensure the scanner modifies the 'dirs' list in-place
        to prevent recursing into .git or __pycache__.
        """
        with patch("pathlib.Path.exists", return_value=True), \
                patch("pathlib.Path.is_dir", return_value=True):
            # Data: Root contains a "good_dir" and a ".git" dir
            # NOTE: We must provide a real list object that can be modified in-place
            dirs_list = ["good_dir", ".git"]
            mock_walk_data = [
                ("/root", dirs_list, [])
            ]

            with patch("app.features.scanner.service.scanner.os.walk", return_value=mock_walk_data):
                list(scanner.scan("/root"))

                # Assert that ".git" was removed from the list in-place
                assert ".git" not in dirs_list
                assert "good_dir" in dirs_list

    def test_scan_permission_error_during_walk(self, scanner):
        """Scenario: os.walk raises PermissionError (e.g., trying to read /root/secret)."""
        with patch("pathlib.Path.exists", return_value=True), \
                patch("pathlib.Path.is_dir", return_value=True):
            # Mock os.walk to crash
            with patch("app.features.scanner.service.scanner.os.walk", side_effect=PermissionError("Boom")):
                # Expect the Mapper to catch it and raise ScanPermissionError
                with pytest.raises(exc.ScanPermissionError):
                    list(scanner.scan("/root"))