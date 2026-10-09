import json
import subprocess

import pytest

from app.features.media_compaction import exceptions as exc
from app.features.media_compaction.utils.error_mapper import map_ffmpeg_error


class TestMapFfmpegError:
    @pytest.mark.parametrize("raw, expected, text", [
        (FileNotFoundError("ffprobe"), exc.FFmpegNotFoundError, "not found on the system path"),
        (subprocess.TimeoutExpired(cmd="ffmpeg", timeout=21600), exc.FFmpegTimeoutError, "after 21600 seconds"),
        (subprocess.CalledProcessError(1, "ffmpeg", stderr="moov atom not found"), exc.FFmpegExecutionError,
         "moov atom not found"),
        (json.JSONDecodeError("Expecting value", "", 0), exc.ProbeParseError, "ffprobe's output"),
        (OSError("No space left on device"), exc.MediaCompactionError, "System I/O error"),
        (RuntimeError("cosmic ray"), exc.MediaCompactionError, "unexpected error"),
    ])
    def test_maps_each_failure_to_its_typed_exception(self, raw, expected, text):
        mapped = map_ffmpeg_error(raw)

        assert type(mapped) is expected
        assert text in str(mapped)
        assert mapped.original_exception is raw

    def test_a_typed_exception_passes_through_unchanged(self):
        typed = exc.FFmpegTimeoutError("already mapped")

        assert map_ffmpeg_error(typed) is typed
