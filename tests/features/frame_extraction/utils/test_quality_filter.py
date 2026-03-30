# Canvas: test_quality_filter_v1
# Path: tests/features/frame_extraction/utils/test_quality_filter.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.features.frame_extraction.utils.quality_filter import QualityFilter


class TestQualityFilter:
    @pytest.fixture
    def quality_filter(self):
        return QualityFilter(blur_threshold=100.0)

    @patch("app.features.frame_extraction.utils.quality_filter.cv2.Laplacian")
    @patch("app.features.frame_extraction.utils.quality_filter.cv2.cvtColor")
    @patch("app.features.frame_extraction.utils.quality_filter.cv2.imread")
    @patch("pathlib.Path.exists", return_value=True)
    def test_filter_frames_keeps_sharp_frames(self, mock_exists, mock_imread, mock_cvt, mock_laplacian, quality_filter):
        mock_imread.return_value = MagicMock()
        mock_cvt.return_value = MagicMock()

        # Mock a high variance (sharp image)
        mock_lap_var = MagicMock()
        mock_lap_var.var.return_value = 150.0
        mock_laplacian.return_value = mock_lap_var

        frames = [(1.0, Path("frame_1.jpg"))]
        result = quality_filter.filter_frames(frames)

        assert len(result) == 1
        assert result[0][0] == 1.0
        assert result[0][2] == 150.0

    @patch("pathlib.Path.unlink")
    @patch("app.features.frame_extraction.utils.quality_filter.cv2.Laplacian")
    @patch("app.features.frame_extraction.utils.quality_filter.cv2.cvtColor")
    @patch("app.features.frame_extraction.utils.quality_filter.cv2.imread")
    @patch("pathlib.Path.exists", return_value=True)
    def test_filter_frames_drops_blurry_frames(self, mock_exists, mock_imread, mock_cvt, mock_laplacian, mock_unlink,
                                               quality_filter):
        mock_imread.return_value = MagicMock()
        mock_cvt.return_value = MagicMock()

        # Mock a low variance (blurry image)
        mock_lap_var = MagicMock()
        mock_lap_var.var.return_value = 50.0
        mock_laplacian.return_value = mock_lap_var

        frames = [(1.0, Path("frame_1.jpg"))]
        result = quality_filter.filter_frames(frames)

        assert len(result) == 0
        mock_unlink.assert_called_once()

    @patch("pathlib.Path.unlink")
    @patch("app.features.frame_extraction.utils.quality_filter.cv2.imread", return_value=None)
    @patch("pathlib.Path.exists", return_value=True)
    def test_graceful_degradation_on_corrupt_image(self, mock_exists, mock_imread, mock_unlink, quality_filter):
        frames = [(1.0, Path("corrupt.jpg"))]
        result = quality_filter.filter_frames(frames)

        # Ensure it didn't crash and cleaned up the bad file
        assert len(result) == 0
        mock_unlink.assert_called_once()