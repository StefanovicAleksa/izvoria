import logging
import uuid
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.features.media_compaction.domain.entities import CompactionOutcome, CompactionStatus, MediaProbe
from app.features.media_compaction.domain.interfaces import IMediaTranscoder
from app.features.media_compaction.exceptions.base import MediaCompactionError
from app.features.media_compaction.utils.error_mapper import map_ffmpeg_error
from app.features.media_compaction.utils.ffmpeg_args import is_compact_video
from app.features.storage.domain.entities import SourceType
from app.features.storage.service.storage_api import StorageAPI

logger = logging.getLogger(__name__)

# A file a browser already plays that is already this lean is left alone: re-encoding it costs CPU and a
# generation of quality for little. 720p H.264 from YouTube is typically 1-2.5 Mbit/s.
_PLAYABLE_AUDIO = {".m4a": {"aac"}, ".mp3": {"mp3"}, ".opus": {"opus"}, ".ogg": {"opus", "vorbis"}}
# How far the copy's length may drift from the original's before it is refused: chunk timestamps were taken
# from the original's timeline (its WAV, decoded from t=0), and a citation seeks the copy to them.
DURATION_TOLERANCE_SEC = 0.5
DURATION_TOLERANCE_RATIO = 0.002


class CompactionAPI:
    """
    Replaces an indexed recording's original with a smaller copy a browser plays as it is. Run only once a
    source is indexed: ingestion reads the full-quality upload, and the copy is what is kept and played back.
    """

    def __init__(self, transcoder: IMediaTranscoder, storage_api: StorageAPI):
        self._transcoder = transcoder
        self._storage = storage_api

    def compact_source(self, source_id: uuid.UUID, dry_run: bool = False) -> CompactionOutcome:
        """`dry_run` only says whether the original would be re-encoded (PENDING), and its current size."""
        source = self._storage.get_source(source_id)
        if not source:
            raise MediaCompactionError(f"Cannot compact: Source {source_id} not found.")
        if source.source_type not in (SourceType.VIDEO, SourceType.AUDIO):
            return CompactionOutcome(source_id=source_id, status=CompactionStatus.NOT_MEDIA)

        try:
            original = self._storage.get_file_path(source.file_id)
        except Exception as e:
            raise MediaCompactionError(f"Cannot compact Source {source_id}: {e}", original_exception=e) from e
        before = original.stat().st_size
        probe = self._transcoder.probe(original)

        as_video = source.source_type is SourceType.VIDEO and probe.video_codec is not None
        if self._already_compact(original, probe, as_video):
            return CompactionOutcome(source_id=source_id, status=CompactionStatus.ALREADY_COMPACT,
                                     bytes_before=before, bytes_after=before)
        if dry_run:
            return CompactionOutcome(source_id=source_id, status=CompactionStatus.PENDING, bytes_before=before,
                                     bytes_after=before)

        # Next to the original, so the swap into the vault is a rename on the same filesystem
        copy = original.parent / f".compact-{uuid.uuid4().hex}{'.mp4' if as_video else '.m4a'}"
        try:
            if as_video:
                self._transcoder.compact_video(original, copy, settings.COMPACT_VIDEO_MAX_HEIGHT,
                                               settings.COMPACT_VIDEO_CRF)
            else:
                self._transcoder.compact_audio(original, copy, settings.COMPACT_AUDIO_BITRATE_KBPS)
            after = copy.stat().st_size
            refusal = self._refusal(probe, self._transcoder.probe(copy), before, after)
            if refusal:
                logger.info(f"Keeping the original of {source.name}: {refusal}")
                return CompactionOutcome(source_id=source_id, status=CompactionStatus.KEPT_ORIGINAL,
                                         bytes_before=before, bytes_after=before, reason=refusal)
            self._storage.replace_file(source.file_id, copy)
            logger.info(f"Compacted {source.name}: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
            return CompactionOutcome(source_id=source_id, status=CompactionStatus.REPLACED,
                                     bytes_before=before, bytes_after=after)
        except MediaCompactionError:
            raise
        except Exception as e:
            raise map_ffmpeg_error(e) from e
        finally:
            copy.unlink(missing_ok=True)

    @staticmethod
    def _already_compact(path: Path, probe: MediaProbe, as_video: bool) -> bool:
        suffix = path.suffix.lower()
        if as_video:
            # The same rule the hosted VPS's video preparer uses, so its 720p copies are never re-encoded here
            return is_compact_video(suffix, probe.video_codec, probe.audio_codec, probe.width, probe.height,
                                    probe.bit_rate, settings.COMPACT_VIDEO_MAX_HEIGHT)
        target = settings.COMPACT_AUDIO_BITRATE_KBPS * 1000
        return (probe.audio_codec in _PLAYABLE_AUDIO.get(suffix, ())
                and probe.audio_bit_rate is not None and probe.audio_bit_rate <= 2 * target)

    @staticmethod
    def _refusal(original: MediaProbe, copy: MediaProbe, before: int, after: int) -> Optional[str]:
        if original.duration_sec is None or copy.duration_sec is None:
            return "its length could not be read, so the copy's timeline could not be checked"
        tolerance = max(DURATION_TOLERANCE_SEC, original.duration_sec * DURATION_TOLERANCE_RATIO)
        if abs(original.duration_sec - copy.duration_sec) > tolerance:
            return (f"the copy runs {copy.duration_sec:.2f}s against the original's {original.duration_sec:.2f}s, "
                    f"so citations would seek to the wrong moment")
        if after > before * (1 - settings.COMPACT_MIN_SAVING):
            return f"the copy saves too little ({before / 1e6:.1f} MB -> {after / 1e6:.1f} MB)"
        return None
