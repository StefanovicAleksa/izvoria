import pytest
import os
from app.features.scanner.service.scanner import FileScanner
from app.features.scanner import exceptions as exc


class TestScannerIntegration:

    @pytest.fixture
    def scanner(self):
        """Creates a real instance of the FileScanner."""
        return FileScanner()

    @pytest.fixture
    def vault_structure(self, tmp_path):
        """
        Creates a realistic nested file structure in a temp directory.
        /root
          ├── video.mp4          (Valid)
          ├── notes.txt          (Valid)
          ├── script.py          (Invalid Extension)
          ├── .git/              (Ignored Dir)
          │    └── config        (Should be skipped)
          └── subfolder/
               └── audio.mp3     (Valid)
        """
        root = tmp_path / "root"
        root.mkdir()

        # 1. Create Valid Files
        (root / "video.mp4").write_text("fake video content")
        (root / "notes.txt").write_text("fake text content")

        # 2. Create Invalid Files (Extension not allowed)
        (root / "script.py").write_text("print('hello')")

        # 3. Create Nested Structure
        sub = root / "subfolder"
        sub.mkdir()
        (sub / "audio.mp3").write_text("fake audio")

        # 4. Create Ignored Directory (e.g., .git)
        git = root / ".git"
        git.mkdir()
        (git / "config").write_text("git config")

        return root

    def test_end_to_end_scan(self, scanner, vault_structure):
        """
        Verifies that the scanner correctly traverses real folders,
        filters extensions, and ignores blocklisted directories.
        """
        # Execute the scan on the real temp directory
        results = list(scanner.scan(vault_structure))

        # Sort by file_name to make assertions deterministic
        results.sort(key=lambda x: x.file_name)

        # Assertions
        assert len(results) == 3

        # Check filenames
        filenames = [r.file_name for r in results]
        assert "audio.mp3" in filenames
        assert "notes.txt" in filenames
        assert "video.mp4" in filenames

        # Verify Filtering
        assert "script.py" not in filenames
        assert "config" not in filenames  # Should not enter .git

        # Verify Data Integrity (Check one file detail)
        # We find the specific file object to check its properties
        audio_file = next(r for r in results if r.file_name == "audio.mp3")
        assert audio_file.extension == ".mp3"
        assert audio_file.size_bytes == 10  # len("fake audio")
        # Verify it has a valid, absolute Path
        assert audio_file.path.is_absolute()
        assert audio_file.path.exists()

    def test_scan_non_existent_path(self, scanner, tmp_path):
        """Ensure integration fails gracefully if path is missing."""
        missing_path = tmp_path / "ghost_dir"

        with pytest.raises(exc.ScanPathNotFound):
            list(scanner.scan(missing_path))

    def test_scan_permission_denied_integration(self, scanner, tmp_path):
        """
        Real permission test.
        Creates a locked directory and ensures the scanner raises ScanPermissionError.
        """
        # Skip if running as root (Docker default), as root ignores permissions
        if hasattr(os, "getuid") and os.getuid() == 0:
            pytest.skip("Skipping permission test: Running as root (permissions ignored)")

        # Create a directory
        locked_dir = tmp_path / "locked_data"
        locked_dir.mkdir()
        (locked_dir / "secret.txt").write_text("secret")

        # Remove read/execute permissions (chmod 000)
        original_mode = locked_dir.stat().st_mode
        locked_dir.chmod(0o000)

        try:
            # Attempt scan
            with pytest.raises(exc.ScanPermissionError):
                list(scanner.scan(locked_dir))

        finally:
            # CLEANUP: Restore permissions so pytest can delete the temp folder
            # If we don't do this, the test runner will crash during cleanup
            locked_dir.chmod(original_mode)