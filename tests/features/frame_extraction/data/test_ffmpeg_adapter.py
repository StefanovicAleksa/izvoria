# Canvas: test_frame_ffmpeg_adapter_v3
# Path: tests/features/frame_extraction/data/test_ffmpeg_adapter.py

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch

from app.core.config import settings
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.data.ffmpeg_adapter import FFmpegExtractionAdapter

RUN_PATH = "app.features.frame_extraction.data.ffmpeg_adapter.subprocess.run"


def _writes_frames(output_dir: Path, count: int):
    """subprocess.run stand-in that writes `count` numbered JPEGs, like the fps filter would."""
    def _run(cmd, **kwargs):
        for index in range(count):
            (output_dir / f"frame_{index:05d}.jpg").write_bytes(b"jpg")
        return subprocess.CompletedProcess(args=cmd, returncode=0)
    return _run


class TestFFmpegExtractionAdapter:
    @pytest.fixture
    def adapter(self):
        return FFmpegExtractionAdapter()

    def test_extract_raw_frames_decodes_the_chunk_in_one_process(self, adapter, tmp_path):
        chunk = ContextualizedChunk(start_sec=10.0, end_sec=13.0)

        with patch(RUN_PATH, side_effect=_writes_frames(tmp_path, 3)) as mock_run:
            frames = adapter.extract_raw_frames("video.mp4", tmp_path, chunk, fps=1)

        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == [
            "ffmpeg", "-nostdin", "-y",
            "-err_detect", "ignore_err", "-fflags", "+discardcorrupt",
            "-ss", "10.000",
            "-i", "video.mp4",
            "-t", "3.000",
            "-vf", "fps=1",
            "-q:v", "2",
            "-threads", str(settings.FRAME_FFMPEG_THREADS),
            "-start_number", "0",
            str(tmp_path / "frame_%05d.jpg")
        ]
        assert mock_run.call_args.kwargs["timeout"] == 60.0
        assert mock_run.call_args.kwargs["stdin"] == subprocess.DEVNULL
        assert frames == [
            (10.0, tmp_path / "frame_00000.jpg"),
            (11.0, tmp_path / "frame_00001.jpg"),
            (12.0, tmp_path / "frame_00002.jpg"),
        ]

    def test_extract_raw_frames_drops_frames_at_or_past_the_chunk_end(self, adapter, tmp_path):
        chunk = ContextualizedChunk(start_sec=0.0, end_sec=2.0)

        with patch(RUN_PATH, side_effect=_writes_frames(tmp_path, 3)):
            frames = adapter.extract_raw_frames("video.mp4", tmp_path, chunk, fps=1)

        assert [timestamp for timestamp, _ in frames] == [0.0, 1.0]

    def test_extract_raw_frames_keeps_frames_written_before_a_failure(self, adapter, tmp_path):
        chunk = ContextualizedChunk(start_sec=5.0, end_sec=8.0)

        def _fails_after_one_frame(cmd, **kwargs):
            (tmp_path / "frame_00000.jpg").write_bytes(b"jpg")
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="corrupt packet")

        with patch(RUN_PATH, side_effect=_fails_after_one_frame) as mock_run:
            frames = adapter.extract_raw_frames("video.mp4", tmp_path, chunk, fps=1)

        assert frames == [(5.0, tmp_path / "frame_00000.jpg")]
        mock_run.assert_called_once()

    def test_extract_raw_frames_falls_back_to_the_midpoint_frame(self, adapter, tmp_path):
        chunk = ContextualizedChunk(start_sec=4.0, end_sec=4.5)

        def _only_single_frame_extraction_writes(cmd, **kwargs):
            if "-vframes" in cmd:
                Path(cmd[-1]).write_bytes(b"jpg")
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        with patch(RUN_PATH, side_effect=_only_single_frame_extraction_writes) as mock_run:
            frames = adapter.extract_raw_frames("video.mp4", tmp_path, chunk, fps=1)

        assert frames == [(4.25, tmp_path / "frame_midpoint.jpg")]
        assert mock_run.call_count == 2
        fallback_cmd = mock_run.call_args_list[1][0][0]
        assert fallback_cmd[fallback_cmd.index("-ss") + 1] == "4.25"

    def test_extract_raw_frames_skips_an_empty_interval(self, adapter, tmp_path):
        chunk = ContextualizedChunk(start_sec=5.0, end_sec=5.0)

        with patch(RUN_PATH) as mock_run:
            frames = adapter.extract_raw_frames("video.mp4", tmp_path, chunk, fps=1)

        assert frames == []
        mock_run.assert_not_called()

    @patch(RUN_PATH)
    def test_extract_frame_success_with_corruption_flags(self, mock_run, adapter):
        """Verifies that the adapter executes FFmpeg with the correct resilience flags."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        result = adapter.extract_frame("corrupted_video.mp4", 10.5, "output.jpg")

        assert result is True
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == [
            "ffmpeg", "-nostdin", "-y",
            "-err_detect", "ignore_err", "-fflags", "+discardcorrupt",
            "-ss", "10.5",
            "-i", "corrupted_video.mp4",
            "-vframes", "1",
            "-q:v", "2",
            "output.jpg"
        ]

    @patch(RUN_PATH)
    def test_extract_frame_handles_called_process_error(self, mock_run, adapter):
        """Verifies that unrecoverable FFmpeg errors return False safely."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="ffmpeg",
            stderr="Fatal unrecoverable error"
        )

        result = adapter.extract_frame("video.mp4", 15.0, "output.jpg")

        assert result is False

    @patch(RUN_PATH)
    def test_extract_frame_handles_timeout(self, mock_run, adapter):
        """Verifies that hanging FFmpeg processes return False safely."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)

        result = adapter.extract_frame("video.mp4", 20.0, "output.jpg")

        assert result is False
