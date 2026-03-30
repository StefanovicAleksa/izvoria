# Canvas: test_frame_deduplicator_v1
# Path: tests/features/frame_extraction/utils/test_frame_deduplicator.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.features.frame_extraction.utils.frame_deduplicator import FrameDeduplicator


class TestFrameDeduplicator:
    @pytest.fixture
    def deduplicator(self):
        return FrameDeduplicator(hamming_threshold=5)

    @patch("app.features.frame_extraction.utils.frame_deduplicator.imagehash.phash")
    @patch("app.features.frame_extraction.utils.frame_deduplicator.Image.open")
    @patch("pathlib.Path.exists", return_value=True)
    def test_deduplicate_keeps_distinct_frames(self, mock_exists, mock_open, mock_phash, deduplicator):
        mock_open.return_value.__enter__.return_value = MagicMock()

        hash1 = MagicMock()
        hash2 = MagicMock()
        hash2.__sub__.return_value = 10  # Distance > threshold
        mock_phash.side_effect = [hash1, hash2]

        frames = [(1.0, Path("f1.jpg"), 100.0), (2.0, Path("f2.jpg"), 110.0)]
        result = deduplicator.deduplicate(frames)

        assert len(result) == 2
        assert result[0][0] == 1.0
        assert result[1][0] == 2.0

    @patch("pathlib.Path.unlink")
    @patch("app.features.frame_extraction.utils.frame_deduplicator.imagehash.phash")
    @patch("app.features.frame_extraction.utils.frame_deduplicator.Image.open")
    @patch("pathlib.Path.exists", return_value=True)
    def test_deduplicate_drops_similar_frames(self, mock_exists, mock_open, mock_phash, mock_unlink, deduplicator):
        mock_open.return_value.__enter__.return_value = MagicMock()

        hash1 = MagicMock()
        hash2 = MagicMock()
        hash2.__sub__.return_value = 2  # Distance <= threshold
        mock_phash.side_effect = [hash1, hash2]

        frames = [(1.0, Path("f1.jpg"), 100.0), (2.0, Path("f2.jpg"), 110.0)]
        result = deduplicator.deduplicate(frames)

        assert len(result) == 1
        assert result[0][0] == 1.0
        mock_unlink.assert_called_once()

    @patch("pathlib.Path.unlink")
    @patch("app.features.frame_extraction.utils.frame_deduplicator.Image.open",
           side_effect=[Exception("Corrupt"), MagicMock()])
    @patch("app.features.frame_extraction.utils.frame_deduplicator.imagehash.phash")
    @patch("pathlib.Path.exists", return_value=True)
    def test_graceful_degradation_on_corrupt_image(self, mock_exists, mock_phash, mock_open, mock_unlink, deduplicator):
        mock_phash.return_value = MagicMock()

        frames = [(1.0, Path("bad.jpg"), 100.0), (2.0, Path("good.jpg"), 110.0)]
        result = deduplicator.deduplicate(frames)

        # Should drop the bad one, keep the good one, and not crash
        assert len(result) == 1
        assert result[0][0] == 2.0
        mock_unlink.assert_called_once()