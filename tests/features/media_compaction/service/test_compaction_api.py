import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.features.media_compaction.domain.entities import CompactionStatus, MediaProbe
from app.features.media_compaction.exceptions import FFmpegExecutionError, MediaCompactionError
from app.features.media_compaction.service import compaction_api as module
from app.features.media_compaction.service.compaction_api import CompactionAPI
from app.features.storage.domain.entities import Source, SourceType

HOUR = 3600.0


@pytest.fixture(autouse=True)
def compaction_settings(monkeypatch):
    monkeypatch.setattr(module.settings, "COMPACT_VIDEO_MAX_HEIGHT", 720)
    monkeypatch.setattr(module.settings, "COMPACT_VIDEO_CRF", 26)
    monkeypatch.setattr(module.settings, "COMPACT_AUDIO_BITRATE_KBPS", 64)
    monkeypatch.setattr(module.settings, "COMPACT_MIN_SAVING", 0.2)


class TestCompactSource:
    @pytest.fixture
    def vault(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def storage(self):
        return MagicMock()

    @pytest.fixture
    def transcoder(self):
        return MagicMock()

    @pytest.fixture
    def api(self, transcoder, storage):
        return CompactionAPI(transcoder=transcoder, storage_api=storage)

    def _source(self, storage, vault, name, source_type, size):
        original = vault / name
        original.write_bytes(b"x" * size)
        source = Source(name=name, source_type=source_type, content_hash="h", file_id=uuid.uuid4())
        storage.get_source.return_value = source
        storage.get_file_path.return_value = original
        return source, original

    @staticmethod
    def _writes(size):
        """A transcoder side effect that writes a copy of `size` bytes where it was told to."""
        def write(_src, dest, *_args):
            Path(dest).write_bytes(b"y" * size)
        return write

    def test_a_1080p_video_is_replaced_by_a_720p_copy(self, api, storage, transcoder, vault):
        source, original = self._source(storage, vault, "talk.mkv", SourceType.VIDEO, 1000)
        transcoder.probe.side_effect = [
            MediaProbe(video_codec="hevc", width=1920, height=1080, audio_codec="aac", duration_sec=HOUR),
            MediaProbe(video_codec="h264", width=1280, height=720, audio_codec="aac", duration_sec=HOUR + 0.04),
        ]
        transcoder.compact_video.side_effect = self._writes(300)
        moved = []
        storage.replace_file.side_effect = lambda file_id, path: moved.append((file_id, Path(path).read_bytes()))

        outcome = api.compact_source(source.id)

        copy = transcoder.compact_video.call_args.args[1]
        transcoder.compact_video.assert_called_once_with(original, copy, 720, 26)
        assert copy.parent == vault and copy.name.startswith(".compact-") and copy.suffix == ".mp4"
        assert moved == [(source.file_id, b"y" * 300)]
        assert (outcome.status, outcome.bytes_before, outcome.bytes_after, outcome.bytes_saved) == (
            CompactionStatus.REPLACED, 1000, 300, 700)
        assert not copy.exists()

    def test_an_audio_recording_becomes_an_aac_m4a(self, api, storage, transcoder, vault):
        source, original = self._source(storage, vault, "call.wav", SourceType.AUDIO, 1000)
        transcoder.probe.side_effect = [MediaProbe(audio_codec="pcm_s16le", audio_bit_rate=1_536_000,
                                                   duration_sec=HOUR),
                                        MediaProbe(audio_codec="aac", duration_sec=HOUR)]
        transcoder.compact_audio.side_effect = self._writes(50)

        outcome = api.compact_source(source.id)

        copy = transcoder.compact_audio.call_args.args[1]
        transcoder.compact_audio.assert_called_once_with(original, copy, 64)
        assert copy.suffix == ".m4a"
        assert outcome.status is CompactionStatus.REPLACED
        transcoder.compact_video.assert_not_called()

    def test_a_video_file_without_a_picture_is_compacted_as_audio(self, api, storage, transcoder, vault):
        source, _ = self._source(storage, vault, "voice.mp4", SourceType.VIDEO, 1000)
        transcoder.probe.side_effect = [MediaProbe(audio_codec="pcm_s16le", duration_sec=HOUR),
                                        MediaProbe(audio_codec="aac", duration_sec=HOUR)]
        transcoder.compact_audio.side_effect = self._writes(50)

        api.compact_source(source.id)

        transcoder.compact_video.assert_not_called()
        transcoder.compact_audio.assert_called_once()

    @pytest.mark.parametrize("name, source_type, probe", [
        ("yt.mp4", SourceType.VIDEO, MediaProbe(video_codec="h264", width=1280, height=720, audio_codec="aac",
                                                bit_rate=1_800_000, duration_sec=HOUR)),
        ("portrait.mp4", SourceType.VIDEO, MediaProbe(video_codec="h264", width=720, height=1280,
                                                      bit_rate=2_000_000, duration_sec=HOUR)),
        ("memo.m4a", SourceType.AUDIO, MediaProbe(audio_codec="aac", audio_bit_rate=96_000, duration_sec=HOUR)),
        ("memo.mp3", SourceType.AUDIO, MediaProbe(audio_codec="mp3", audio_bit_rate=128_000, duration_sec=HOUR)),
    ])
    def test_a_file_that_is_already_lean_and_playable_is_left_alone(self, api, storage, transcoder, vault,
                                                                      name, source_type, probe):
        source, _ = self._source(storage, vault, name, source_type, 1000)
        transcoder.probe.return_value = probe

        outcome = api.compact_source(source.id)

        assert outcome.status is CompactionStatus.ALREADY_COMPACT
        assert outcome.bytes_saved == 0
        transcoder.compact_video.assert_not_called()
        transcoder.compact_audio.assert_not_called()

    @pytest.mark.parametrize("probe", [
        # 720p H.264 but a screen recording at 12 Mbit/s: worth re-encoding
        MediaProbe(video_codec="h264", width=1280, height=720, audio_codec="aac", bit_rate=12_000_000,
                   duration_sec=HOUR),
        # Lean, but in a container the browser won't play
        MediaProbe(video_codec="h264", width=1280, height=720, audio_codec="aac", bit_rate=1_000_000,
                   duration_sec=HOUR),
    ])
    def test_a_heavy_or_unplayable_720p_file_is_still_compacted(self, api, storage, transcoder, vault, probe):
        name = "screen.mp4" if probe.bit_rate > 3_000_000 else "lean.mkv"
        source, _ = self._source(storage, vault, name, SourceType.VIDEO, 1000)
        transcoder.probe.side_effect = [probe, probe.model_copy(update={"bit_rate": 900_000})]
        transcoder.compact_video.side_effect = self._writes(200)

        assert api.compact_source(source.id).status is CompactionStatus.REPLACED

    @pytest.mark.parametrize("copy_probe, copy_size, reason", [
        (MediaProbe(video_codec="h264", duration_sec=HOUR - 12), 300, "seek to the wrong moment"),
        (MediaProbe(video_codec="h264", duration_sec=None), 300, "length could not be read"),
        (MediaProbe(video_codec="h264", duration_sec=HOUR), 850, "saves too little"),
    ])
    def test_a_copy_that_fails_a_guard_is_thrown_away(self, api, storage, transcoder, vault,
                                                      copy_probe, copy_size, reason):
        source, original = self._source(storage, vault, "talk.mkv", SourceType.VIDEO, 1000)
        transcoder.probe.side_effect = [MediaProbe(video_codec="hevc", width=1920, height=1080, duration_sec=HOUR),
                                        copy_probe]
        transcoder.compact_video.side_effect = self._writes(copy_size)

        outcome = api.compact_source(source.id)

        assert outcome.status is CompactionStatus.KEPT_ORIGINAL
        assert reason in outcome.reason
        storage.replace_file.assert_not_called()
        assert original.read_bytes() == b"x" * 1000
        assert [p.name for p in vault.iterdir()] == ["talk.mkv"]

    def test_documents_and_images_are_not_touched(self, api, storage, transcoder):
        source = Source(name="a.pdf", source_type=SourceType.DOCUMENT, content_hash="h", file_id=uuid.uuid4())
        storage.get_source.return_value = source

        assert api.compact_source(source.id).status is CompactionStatus.NOT_MEDIA
        transcoder.probe.assert_not_called()

    def test_a_failed_transcode_raises_and_leaves_no_copy(self, api, storage, transcoder, vault):
        source, _ = self._source(storage, vault, "talk.mkv", SourceType.VIDEO, 1000)
        transcoder.probe.return_value = MediaProbe(video_codec="hevc", width=1920, height=1080, duration_sec=HOUR)
        transcoder.compact_video.side_effect = FFmpegExecutionError("corrupt")

        with pytest.raises(FFmpegExecutionError):
            api.compact_source(source.id)

        storage.replace_file.assert_not_called()
        assert [p.name for p in vault.iterdir()] == ["talk.mkv"]

    def test_a_missing_source_raises(self, api, storage):
        storage.get_source.return_value = None

        with pytest.raises(MediaCompactionError, match="not found"):
            api.compact_source(uuid.uuid4())

    def test_a_missing_original_raises(self, api, storage):
        storage.get_source.return_value = Source(name="a.mp4", source_type=SourceType.VIDEO, content_hash="h",
                                                 file_id=uuid.uuid4())
        storage.get_file_path.side_effect = FileNotFoundError("gone")

        with pytest.raises(MediaCompactionError, match="gone"):
            api.compact_source(uuid.uuid4())

    def test_a_dry_run_says_what_would_be_reencoded_without_touching_it(self, api, storage, transcoder, vault):
        source, _ = self._source(storage, vault, "talk.mkv", SourceType.VIDEO, 1000)
        transcoder.probe.return_value = MediaProbe(video_codec="hevc", width=1920, height=1080, duration_sec=HOUR)

        outcome = api.compact_source(source.id, dry_run=True)

        assert (outcome.status, outcome.bytes_before, outcome.bytes_saved) == (CompactionStatus.PENDING, 1000, 0)
        transcoder.compact_video.assert_not_called()
        storage.replace_file.assert_not_called()
