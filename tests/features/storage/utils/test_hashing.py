import os
import pytest
import tempfile
from app.features.storage.utils.hashing import calculate_file_hash


class TestHashingUtil:
    def test_calculate_file_hash_happy_path(self):
        """
        Should correctly calculate SHA-256 for a known string.
        Verifies the chunking logic works without error.
        """
        # Create a temp file with content "hello world"
        # We use delete=False so we can close it and let the hasher open it again
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("hello world")
            temp_path = f.name

        try:
            # Expected SHA-256 for "hello world" calculated via terminal
            expected_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

            result = calculate_file_hash(temp_path)
            assert result == expected_hash
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_calculate_file_hash_file_not_found(self):
        """
        Should raise FileNotFoundError if the file does not exist.
        """
        with pytest.raises(FileNotFoundError):
            calculate_file_hash("/path/to/non/existent/ghost.file")