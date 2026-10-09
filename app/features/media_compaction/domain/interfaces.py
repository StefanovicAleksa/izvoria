from abc import ABC, abstractmethod
from pathlib import Path

from app.features.media_compaction.domain.entities import MediaProbe


class IMediaTranscoder(ABC):
    """Contract for reading a recording's streams and writing a smaller, browser-playable copy of it."""

    @abstractmethod
    def probe(self, path: str | Path) -> MediaProbe:
        """Reads the codecs, frame size, audio bit rate and duration of a media file."""
        pass

    @abstractmethod
    def compact_video(self, source_path: str | Path, dest_path: str | Path, max_height: int, crf: int) -> None:
        """Writes an H.264/AAC MP4 of the whole recording, its short side at most `max_height`, to `dest_path`."""
        pass

    @abstractmethod
    def compact_audio(self, source_path: str | Path, dest_path: str | Path, bitrate_kbps: int) -> None:
        """Writes an AAC m4a of the whole recording at `bitrate_kbps` to `dest_path`."""
        pass
