# Canvas: test_bm25_repo_v1
# Path: tests/features/rag_storage/data/test_bm25_repo.py

import uuid
from unittest.mock import patch, mock_open

from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_storage.data.bm25_repo import LocalBM25Repository

class TestLocalBM25Repository:
    @patch("os.makedirs")
    def test_save_sparse_index_success(self, mock_makedirs):
        repo = LocalBM25Repository()
        source_id = uuid.uuid4()
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Hello world")

        with patch("builtins.open", mock_open()) as mocked_file:
            with patch("pickle.dump") as mock_pickle:
                result = repo.save_sparse_index(source_id, [chunk])

                assert result == 1
                mocked_file.assert_called_once()
                mock_pickle.assert_called_once()

                # Verify payload structure
                saved_payload = mock_pickle.call_args[0][0]
                assert saved_payload["source_id"] == str(source_id)
                assert saved_payload["chunk_ids"] == [chunk.chunk_id]
                assert "bm25_index" in saved_payload

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_delete_by_source_id_success(self, mock_remove, mock_exists, mock_makedirs):
        repo = LocalBM25Repository()
        repo.delete_by_source_id(uuid.uuid4())
        mock_remove.assert_called_once()