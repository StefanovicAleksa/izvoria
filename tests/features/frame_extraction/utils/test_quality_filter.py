# Canvas: test_quality_filter_v2
# Path: tests/features/frame_extraction/utils/test_quality_filter.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.core.config import settings
from app.features.frame_extraction.utils.quality_filter import QualityFilter

MODULE = "app.features.frame_extraction.utils.quality_filter"


def _laplacians(*variances: float):
    return [MagicMock(**{"var.return_value": variance}) for variance in variances]


class TestQualityFilterConfig:
    def test_default_threshold_comes_from_settings(self):
        assert QualityFilter().blur_threshold == settings.FRAME_BLUR_THRESHOLD

    def test_explicit_threshold_wins(self):
        assert QualityFilter(blur_threshold=42.0).blur_threshold == 42.0


@patch("pathlib.Path.unlink")
@patch(f"{MODULE}.cv2.Laplacian")
@patch(f"{MODULE}.cv2.cvtColor")
@patch(f"{MODULE}.cv2.imread")
@patch("pathlib.Path.exists", return_value=True)
class TestQualityFilter:
    @pytest.fixture
    def quality_filter(self):
        return QualityFilter(blur_threshold=100.0)

    def test_keeps_sharp_frames_and_deletes_blurry_ones(
            self, mock_exists, mock_imread, mock_cvt, mock_laplacian, mock_unlink, quality_filter
    ):
        mock_laplacian.side_effect = _laplacians(150.0, 50.0)

        result = quality_filter.filter_frames([(1.0, Path("a.jpg")), (2.0, Path("b.jpg"))])

        assert result == [(1.0, Path("a.jpg"), 150.0)]
        mock_unlink.assert_called_once_with(missing_ok=True)

    def test_keeps_the_sharpest_frame_when_all_are_blurry(
            self, mock_exists, mock_imread, mock_cvt, mock_laplacian, mock_unlink, quality_filter
    ):
        mock_laplacian.side_effect = _laplacians(40.0, 80.0, 10.0)

        result = quality_filter.filter_frames([(1.0, Path("a.jpg")), (2.0, Path("b.jpg")), (3.0, Path("c.jpg"))])

        assert result == [(2.0, Path("b.jpg"), 80.0)]
        assert mock_unlink.call_count == 2

    def test_never_keeps_a_uniform_frame(
            self, mock_exists, mock_imread, mock_cvt, mock_laplacian, mock_unlink, quality_filter
    ):
        mock_laplacian.side_effect = _laplacians(0.0)

        result = quality_filter.filter_frames([(1.0, Path("black.jpg"))])

        assert result == []
        mock_unlink.assert_called_once_with(missing_ok=True)

    def test_graceful_degradation_on_corrupt_image(
            self, mock_exists, mock_imread, mock_cvt, mock_laplacian, mock_unlink, quality_filter
    ):
        mock_imread.return_value = None

        result = quality_filter.filter_frames([(1.0, Path("corrupt.jpg"))])

        assert result == []
        mock_laplacian.assert_not_called()
        mock_unlink.assert_called_once_with(missing_ok=True)
