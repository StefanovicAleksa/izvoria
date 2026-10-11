# Canvas: test_glossary_miner_file_storage_v3
# Path: tests/features/glossary_miner/data/test_file_storage_adapter.py

import pytest
from unittest.mock import patch, mock_open

from app.tools.glossary_miner.data.file_storage_adapter import LocalFileStorageAdapter
from app.tools.glossary_miner.domain.entities import IzvoriaGlossaryManifest
from app.tools.glossary_miner.exceptions import DataProcessingError


class TestLocalFileStorageAdapter:

    @patch("app.tools.glossary_miner.data.file_storage_adapter.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_manifest_success(self, mock_file_open, mock_mkdir):
        adapter = LocalFileStorageAdapter(base_dir="/fake/dir")
        manifest = IzvoriaGlossaryManifest()

        result = adapter.save_manifest("test_glossary.json", manifest)

        mock_mkdir.assert_called_once()
        mock_file_open.assert_called_once()
        assert result.endswith("test_glossary.json")

        handle = mock_file_open()
        handle.write.assert_called_once()
        written_data = handle.write.call_args[0][0]
        assert "sub_domains" in written_data

    @patch("app.tools.glossary_miner.data.file_storage_adapter.Path.mkdir")
    @patch("builtins.open", side_effect=PermissionError("Access Denied"))
    def test_save_manifest_io_error(self, mock_file_open, mock_mkdir):
        adapter = LocalFileStorageAdapter(base_dir="/fake/dir")
        manifest = IzvoriaGlossaryManifest()

        with pytest.raises(DataProcessingError, match="Failed to write glossary manifest"):
            adapter.save_manifest("test.json", manifest)

        # Assert the mocks to clear PyCharm's 'unused parameter' warnings
        mock_mkdir.assert_called_once()
        mock_file_open.assert_called_once()