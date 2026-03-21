# Canvas: ffmpeg_adapter_v1
# Path: app/features/domain_adaptation/data/ffmpeg_adapter.py

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

from app.features.domain_adaptation.domain.interfaces import IFfmpegAdapter
from app.features.domain_adaptation.domain.entities import TimeSlice, AudioSample
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error

class FfmpegAdapter(IFfmpegAdapter):
    """Executes lightning-fast OS-level audio slicing using FFmpeg."""

    def extract_samples(self, source_audio_path: Path, slices: List[TimeSlice]) -> List[AudioSample]:
        if not source_audio_path.exists():
            raise FileNotFoundError(f"Source audio missing: {source_audio_path}")

        samples: List[AudioSample] = []

        for ts in slices:
            duration = ts.end_sec - ts.start_sec
            if duration <= 0:
                continue

            # Create a secure temporary file
            temp_fd, temp_path_str = tempfile.mkstemp(suffix=".wav", prefix="onyx_scout_slice_")
            os.close(temp_fd)
            out_path = Path(temp_path_str)

            # Fast-seek stream copy (instantaneous)
            command = [
                "ffmpeg",
                "-ss", str(ts.start_sec),
                "-i", os.fspath(source_audio_path),
                "-t", str(duration),
                "-vn",
                "-acodec", "copy",
                "-y",
                os.fspath(out_path)
            ]

            try:
                subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    timeout=30
                )
                samples.append(AudioSample(
                    start_sec=ts.start_sec,
                    end_sec=ts.end_sec,
                    file_path=str(out_path)
                ))
            except Exception as e:
                # Cleanup on failure to prevent temp file leakage
                if out_path.exists():
                    out_path.unlink(missing_ok=True)
                raise map_domain_adaptation_error(e, context=f"FFmpeg slicing at {ts.start_sec}s") from e

        return samples

    def cleanup_samples(self, samples: List[AudioSample]) -> None:
        """Safely purges the temporary OS files after the Scout is finished."""
        for sample in samples:
            path = Path(sample.file_path)
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass