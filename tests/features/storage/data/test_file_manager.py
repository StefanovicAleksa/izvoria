import pytest
from pathlib import Path
from unittest.mock import patch
from app.features.storage.data.file_manager import FileManager


class TestFileManager:
    @pytest.fixture
    def file_manager(self, tmp_path):
        fm = FileManager()
        fm.base_path = tmp_path
        return fm

    def test_generate_path_structure(self, file_manager):
        """Should create paths like YYYY/MM/DD/uuid.ext"""
        path = file_manager.generate_path(".mp4")
        assert path.suffix == ".mp4"
        assert path.parent.exists()
        assert str(file_manager.base_path) in str(path)

    def test_store_file_copy_happy_path(self, file_manager, tmp_path):
        """Should copy file and return size (move=False)."""
        src = tmp_path / "source.txt"
        src.write_text("content")
        dest = file_manager.generate_path(".txt")

        size = file_manager.store_file(src, dest, move=False)

        assert dest.exists()
        assert dest.read_text() == "content"
        assert src.exists()
        assert size == 7

    def test_store_file_move_happy_path(self, file_manager, tmp_path):
        """Should move file and delete original (move=True)."""
        src = tmp_path / "tobe_moved.txt"
        src.write_text("move me")
        dest = file_manager.generate_path(".txt")

        file_manager.store_file(src, dest, move=True)

        assert dest.exists()
        assert not src.exists()

    def test_store_file_os_error(self, file_manager):
        """Should raise OSError if filesystem fails (simulated)."""
        # FIX: Mock Path.exists so we get past the existence check
        with patch("pathlib.Path.exists", return_value=True):
            with patch("shutil.copy2", side_effect=OSError("Disk full")):
                with pytest.raises(OSError, match="Storage operation failed"):
                    file_manager.store_file("dummy_src", Path("dest"), move=False)