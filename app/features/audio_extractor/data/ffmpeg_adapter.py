# Canvas: audio_ffmpeg_adapter_v2
# Path: app/features/audio_extractor/data/ffmpeg_adapter.py

import os
import subprocess
import tempfile
import wave
import struct
import logging
from pathlib import Path

from app.features.audio_extractor.domain.interfaces import IFFmpegAdapter
from app.features.audio_extractor.utils.error_mapper import map_ffmpeg_error

logger = logging.getLogger(__name__)


class FFmpegAdapter(IFFmpegAdapter):
    def __init__(self, timeout_seconds: int = 3600):
        self._timeout = timeout_seconds

    def extract_audio(self, source_path: str | Path) -> Path:
        """
        Executes FFmpeg to extract audio, managing the temporary OS file securely.
        If the video lacks an audio stream, generates a silent track as a fallback.
        """
        src = Path(source_path).resolve()

        if not src.exists():
            raw_error = FileNotFoundError(f"Source file not found at path: {src}")
            raise map_ffmpeg_error(raw_error)

        # 1. Manage the OS-level temporary file securely
        temp_fd, temp_path_str = tempfile.mkstemp(suffix=".wav", prefix="onyx_audio_")
        os.close(temp_fd)
        temp_path = Path(temp_path_str)

        # Explicitly typing as list[str] and using os.fspath() for Python < 3.12 Windows compatibility
        command: list[str] = [
            "ffmpeg",
            "-i", os.fspath(src),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            os.fspath(temp_path)
        ]

        try:
            # 2. Execute extraction
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=self._timeout
            )
            return temp_path

        except subprocess.CalledProcessError as e:
            # FIX 1: Handle videos with no audio tracks gracefully
            if "does not contain any stream" in e.stderr or "Invalid argument" in e.stderr:
                logger.warning(f"Video {src.name} lacks an audio stream. Generating silent track to preserve pipeline integrity.")
                self._generate_silent_wav(temp_path)
                return temp_path
            else:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                raise map_ffmpeg_error(e) from e

        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise map_ffmpeg_error(e) from e

    def _generate_silent_wav(self, output_path: Path) -> None:
        """Generates a 1-second silent WAV file (16kHz, Mono, 16-bit PCM)."""
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            # Write 16,000 frames of value '0' (1 second of pure silence)
            wav_file.writeframes(struct.pack('<h', 0) * 16000)