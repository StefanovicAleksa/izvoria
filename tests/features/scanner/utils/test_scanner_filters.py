import pytest
from pathlib import Path
from app.features.scanner.utils import filters
from app.features.scanner.utils.error_mapper import map_scanner_error
from app.features.scanner import exceptions as exc

class TestScannerFilters:

    @pytest.mark.parametrize("filename, expected", [
        ("video.mp4", True),
        ("movie.mkv", True),
        ("song.mp3", True),
        ("notes.txt", True),
        ("report.pdf", True),
        ("script.py", False),  # Code not allowed
        ("program.exe", False),  # Binary not allowed
        ("image.jpg", False),  # Images (handled by different pipeline usually) - adjust based on requirements
        (".env", False),  # Config files not allowed
    ])
    def test_is_extension_allowed(self, filename, expected):
        """Verify the whitelist of allowed extensions."""
        # We mock Path just to provide a suffix
        path = Path(f"/tmp/{filename}")
        assert filters.is_extension_allowed(path) is expected

    @pytest.mark.parametrize("dirname, expected", [
        (".git", True),
        ("__pycache__", True),
        ("venv", True),
        ("node_modules", True),
        ("my_videos", False),
        ("documents", False),
        ("Downloads", False),
    ])
    def test_is_directory_ignored(self, dirname, expected):
        """Verify the blacklist of ignored directories."""
        assert filters.is_directory_ignored(dirname) is expected


class TestScannerErrorMapper:

    def test_map_permission_error(self):
        """PermissionError -> ScanPermissionError"""
        raw_err = PermissionError("Access Denied")
        mapped = map_scanner_error(raw_err, context="/root")

        assert isinstance(mapped, exc.ScanPermissionError)
        assert "/root" in str(mapped)

    def test_map_file_not_found_error(self):
        """FileNotFoundError -> ScanPathNotFound"""
        raw_err = FileNotFoundError("Gone")
        mapped = map_scanner_error(raw_err, context="/missing")

        assert isinstance(mapped, exc.ScanPathNotFound)

    def test_map_generic_os_error(self):
        """OSError -> ScannerError"""
        raw_err = OSError("Disk Full")
        mapped = map_scanner_error(raw_err, context="/disk")

        assert isinstance(mapped, exc.ScannerError)
        # Ensure it's not one of the specific subclasses if we didn't want it to be
        assert not isinstance(mapped, exc.ScanPathNotFound)

    def test_map_unknown_error(self):
        """Exception -> ScannerError"""
        raw_err = Exception("Random Crash")
        mapped = map_scanner_error(raw_err)

        assert isinstance(mapped, exc.ScannerError)
        assert "Unexpected scanner error" in str(mapped)