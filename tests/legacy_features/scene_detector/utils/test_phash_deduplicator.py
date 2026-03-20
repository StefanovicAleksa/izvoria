# Canvas: test_phash_deduplicator_v3
# Path: tests/features/scene_detector/test_phash_deduplicator.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.legacy_features.scene_detector.utils.phash_deduplicator import PhashDeduplicator


class TestPhashDeduplicator:
    @pytest.fixture
    def deduplicator(self):
        return PhashDeduplicator(hamming_distance_threshold=5)

    @patch("app.legacy_features.scene_detector.utils.phash_deduplicator.Image.open")
    @patch("app.legacy_features.scene_detector.utils.phash_deduplicator.imagehash.phash")
    def test_deduplicate_all_unique(self, mock_phash, mock_image_open, deduplicator):
        """Verifies frames with Hamming distance > 5 are all kept."""
        mock_image_open.return_value.__enter__.return_value = MagicMock()

        # Hash 1 and Hash 2 are very different (distance 10)
        hash1 = MagicMock()
        hash2 = MagicMock()
        hash2.__sub__.return_value = 10
        mock_phash.side_effect = [hash1, hash2]

        scenes = [(1.0, Path("frame_1.jpg")), (2.0, Path("frame_2.jpg"))]

        result = deduplicator.deduplicate(scenes)

        assert len(result) == 2
        assert result[0][0] == 1.0
        assert result[1][0] == 2.0

    @patch("app.legacy_features.scene_detector.utils.phash_deduplicator.Path.unlink")
    @patch("app.legacy_features.scene_detector.utils.phash_deduplicator.Image.open")
    @patch("app.legacy_features.scene_detector.utils.phash_deduplicator.imagehash.phash")
    def test_deduplicate_removes_duplicates(self, mock_phash, mock_image_open, mock_unlink, deduplicator):
        """Verifies frames with Hamming distance <= 5 are dropped and deleted."""
        mock_image_open.return_value.__enter__.return_value = MagicMock()

        # Hash 1 and Hash 2 are identical (distance 0)
        hash1 = MagicMock()
        hash2 = MagicMock()
        hash2.__sub__.return_value = 0
        mock_phash.side_effect = [hash1, hash2]

        path1 = Path("frame_1.jpg")
        path2 = Path("frame_2.jpg")
        scenes = [(1.0, path1), (2.0, path2)]

        result = deduplicator.deduplicate(scenes)

        assert len(result) == 1
        assert result[0][0] == 1.0
        # The second file should have been physically deleted
        mock_unlink.assert_called_once()

    @patch("app.legacy_features.scene_detector.utils.phash_deduplicator.Path.unlink")
    @patch("app.legacy_features.scene_detector.utils.phash_deduplicator.Image.open")
    @patch("app.legacy_features.scene_detector.utils.phash_deduplicator.imagehash.phash")
    def test_graceful_degradation_on_corrupt_image(self, mock_phash, mock_image_open, mock_unlink, deduplicator):
        """Verifies a corrupted image skips to the next without crashing."""
        # Simulating PIL throwing an error on the first image, but succeeding on the second
        mock_image_open.side_effect = [Exception("Corrupt image file"), MagicMock()]

        # Mock the hash generation for the surviving image to prevent internal math crashes
        mock_phash.return_value = "valid_hash"

        path1 = Path("bad_frame.jpg")
        path2 = Path("good_frame.jpg")
        scenes = [(1.0, path1), (2.0, path2)]

        # Should not crash!
        result = deduplicator.deduplicate(scenes)

        assert len(result) == 1
        assert result[0][0] == 2.0
        # The corrupted file should be deleted to prevent buildup
        mock_unlink.assert_called_once()